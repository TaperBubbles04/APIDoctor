"""Shared logic for the CLI and the web page: diagnose many records and summarise them."""
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from apidoctor.config import Config
from apidoctor.engine import diagnose
from apidoctor.llm.limited import LimitedCachedProvider
from apidoctor.loader import load_logs_text
from apidoctor.models import Diagnosis, LogRecord
from apidoctor.spec import load_spec_text, operation_name

UNKNOWN_OPERATION = "unknown operation"
MAX_BODY_BYTES = 2 * 1024 * 1024   # web request size limit
MAX_LOG_LINES = 1000               # web request line limit


@dataclass
class RecordResult:
    record: LogRecord
    operation: str
    diagnosis: Diagnosis

    def to_dict(self) -> dict:
        r, d = self.record, self.diagnosis
        return {
            "timestamp": r.timestamp, "method": r.method, "url": r.url, "status_code": r.status_code,
            "operation": self.operation, "category": d.category, "retryable": d.retryable,
            "confidence": d.confidence, "source": d.source, "explanation": d.explanation,
            "suggested_fix": d.suggested_fix,
            "spec_reference": {"operation": d.spec_reference.operation, "field": d.spec_reference.field},
        }


@dataclass
class GroupSummary:
    operation: str
    category: str
    count: int
    retryable: bool | None   # None means the diagnoses in this group disagree ("mixed")

    def to_dict(self) -> dict:
        return {"operation": self.operation, "category": self.category,
                "count": self.count, "retryable": self.retryable}


@dataclass
class DiagnoseResult:
    total_records: int = 0
    non_failures: int = 0     # records the rules called "not a failure"
    llm_skipped: int = 0      # records not sent to the LLM because the call limit was reached
    results: list[RecordResult] = field(default_factory=list)
    groups: list[GroupSummary] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_records": self.total_records, "failures": len(self.results),
            "non_failures": self.non_failures, "llm_skipped": self.llm_skipped,
            "groups": [g.to_dict() for g in self.groups],
            "results": [r.to_dict() for r in self.results],
        }


def summarize(results: list[RecordResult]) -> list[GroupSummary]:
    """Group by (operation, category); biggest groups first, ties in alphabetical order."""
    buckets: dict[tuple[str, str], list[bool]] = {}
    for r in results:
        buckets.setdefault((r.operation, r.diagnosis.category), []).append(r.diagnosis.retryable)
    groups = [
        GroupSummary(op, category, len(flags), flags[0] if len(set(flags)) == 1 else None)
        for (op, category), flags in buckets.items()
    ]
    return sorted(groups, key=lambda g: (-g.count, g.operation, g.category))


def diagnose_logs(records: list[LogRecord], spec: dict | None, provider, config: Config,
                  sleep: Callable[[float], None] = time.sleep) -> DiagnoseResult:
    out = DiagnoseResult(total_records=len(records))
    for record in records:
        if provider is not None and hasattr(provider, "refused"):
            provider.refused = False
        diagnosis = diagnose(record, spec, provider, config, sleep)
        if provider is not None and getattr(provider, "refused", False):
            out.llm_skipped += 1
        if diagnosis is None:
            out.non_failures += 1
            continue
        op = (operation_name(spec, record.method, record.url) if spec is not None else None) or UNKNOWN_OPERATION
        out.results.append(RecordResult(record, op, diagnosis))
    out.groups = summarize(out.results)
    return out


def build_report(result: DiagnoseResult, load_errors: list, provider) -> dict:
    """The JSON-safe report shared by the CLI (--json) and the web page."""
    data = result.to_dict()
    data["load_errors"] = [{"line": e.line_number, "message": e.message} for e in load_errors]
    data["llm"] = {"real_calls": provider.real_calls, "cache_hits": provider.cache_hits} if provider else None
    return data


class RequestError(Exception):
    """A problem with a web request. The message is a fixed, safe sentence (never raw input or errors)."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


def parse_payload(raw: bytes) -> dict:
    """Validate the body of POST /api/diagnose: {logs_text, spec_text, use_llm}."""
    if len(raw) > MAX_BODY_BYTES:
        raise RequestError(413, "request body too large (limit 2 MB)")
    try:
        data = json.loads(raw)
    except ValueError:
        raise RequestError(400, "body must be valid JSON") from None
    if not isinstance(data, dict):
        raise RequestError(400, "body must be a JSON object")
    logs, spec, use_llm = data.get("logs_text"), data.get("spec_text"), data.get("use_llm", False)
    if not isinstance(logs, str) or not isinstance(spec, str):
        raise RequestError(400, "logs_text and spec_text must be strings")
    if not isinstance(use_llm, bool):
        raise RequestError(400, "use_llm must be true or false")
    if not logs.strip():
        raise RequestError(400, "logs_text is empty")
    if len(logs.splitlines()) > MAX_LOG_LINES:
        raise RequestError(413, f"too many log lines (limit {MAX_LOG_LINES})")
    return {"logs_text": logs, "spec_text": spec, "use_llm": use_llm}


def diagnose_request(payload: dict, config: Config, inner_provider, cache_path: Path,
                     sleep: Callable[[float], None] = time.sleep) -> dict:
    """Run one web request. inner_provider is get_provider(config): None means no LLM key."""
    try:
        spec = load_spec_text(payload["spec_text"])
    except ValueError:
        raise RequestError(400, "spec is not valid JSON or YAML") from None
    records, load_errors = load_logs_text(payload["logs_text"], config)
    provider = None
    if payload["use_llm"]:
        if inner_provider is None:
            raise RequestError(400, "no LLM key is configured on the server")
        try:   # the call cap per request comes from config
            provider = LimitedCachedProvider(inner_provider, config, cache_path,
                                             config.max_llm_calls_per_run, sleep)
        except ValueError:
            raise RequestError(500, "the LLM cache file is unusable") from None
    result = diagnose_logs(records, spec, provider, config, sleep)
    return build_report(result, load_errors, provider)
