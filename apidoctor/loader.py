"""Read a JSON-lines log file into redacted LogRecord objects."""
import json
from dataclasses import dataclass
from pathlib import Path

from apidoctor.config import Config
from apidoctor.models import LogRecord
from apidoctor.redact import redact_body, redact_headers, redact_url

REQUIRED = ("timestamp", "method", "url", "status_code", "duration_ms")


@dataclass
class LoadError:
    line_number: int
    message: str  # never contains the line's content (it may hold secrets)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check(obj) -> str | None:
    """Return a problem description, or None if the object is a valid record."""
    if not isinstance(obj, dict):
        return "line is not a JSON object"
    for name in REQUIRED:
        if name not in obj:
            return f"missing field '{name}'"
    for name in ("timestamp", "method", "url"):
        if not isinstance(obj[name], str):
            return f"field '{name}' must be a string"
    status = obj["status_code"]
    if status is not None and (not isinstance(status, int) or isinstance(status, bool)):
        return "field 'status_code' must be an integer or null"
    if not _is_number(obj["duration_ms"]) or obj["duration_ms"] < 0:
        return "field 'duration_ms' must be a non-negative number"
    for name in ("request_headers", "response_headers"):
        headers = obj.get(name, {})
        if not isinstance(headers, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in headers.items()
        ):
            return f"field '{name}' must be an object of strings"
    return None


def parse_record(obj, config: Config) -> tuple[LogRecord | None, str | None]:
    """Validate one parsed JSON object and build a redacted LogRecord: (record, None) or (None, problem)."""
    problem = _check(obj)
    if problem:
        return None, problem
    n = config.redaction_min_token_length
    return LogRecord(
        timestamp=obj["timestamp"],
        method=obj["method"],
        url=redact_url(obj["url"], n),
        status_code=obj["status_code"],
        request_headers=redact_headers(obj.get("request_headers", {}), n),
        request_body=redact_body(obj.get("request_body"), n),
        response_body=redact_body(obj.get("response_body"), n),
        duration_ms=obj["duration_ms"],
        response_headers=redact_headers(obj.get("response_headers", {}), n),
    ), None


def _parse_lines(lines, config: Config) -> tuple[list[LogRecord], list[LoadError]]:
    records: list[LogRecord] = []
    errors: list[LoadError] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            # e.msg / e.colno only: never the line's content.
            errors.append(LoadError(line_number, f"invalid JSON ({e.msg} at column {e.colno})"))
            continue
        record, problem = parse_record(obj, config)
        if problem:
            errors.append(LoadError(line_number, problem))
        else:
            records.append(record)
    return records, errors


def load_logs(path: Path, config: Config) -> tuple[list[LogRecord], list[LoadError]]:
    with open(path, encoding="utf-8", errors="replace") as f:
        return _parse_lines(f, config)


def load_logs_text(text: str, config: Config) -> tuple[list[LogRecord], list[LoadError]]:
    """Same as load_logs, for log text that is already in memory (the web page)."""
    return _parse_lines(text.splitlines(), config)
