"""The rules layer: clear cases are diagnosed here, with no LLM. First match wins."""
import json
from dataclasses import dataclass
from typing import Any

from apidoctor.models import Diagnosis, LogRecord, SpecReference
from apidoctor.spec import (
    find_operation, operation_name, required_fields_missing, type_mismatches,
)

# All confidence numbers live here. These are untuned starting guesses.
CONFIDENCE = {
    "network_timeout": 0.85,
    "rate_limit_with_header": 0.95,
    "rate_limit_no_header": 0.80,
    "validation_from_spec": 0.90,
    "validation_from_body": 0.65,
    "auth": 0.85,
    "not_found": 0.80,
    "server_error": 0.80,
}

# Response-body keys that may name the offending field (top level, or inside an
# "errors" / "details" list of objects).
FIELD_KEYS = ("field", "param", "parameter", "property")


@dataclass
class RuleResult:
    outcome: str                          # "diagnosed" | "not_a_failure" | "needs_llm"
    diagnosis: Diagnosis | None = None
    reason: str = ""                      # why it needs the LLM (for needs_llm)


def _diagnosed(category, retryable, confidence_key, explanation, fix, ref=None) -> RuleResult:
    return RuleResult("diagnosed", Diagnosis(
        category=category, retryable=retryable, confidence=CONFIDENCE[confidence_key],
        explanation=explanation, suggested_fix=fix, source="rule",
        spec_reference=ref or SpecReference(),
    ))


def _needs_llm(reason: str) -> RuleResult:
    return RuleResult("needs_llm", None, reason)


def _as_json(body: Any) -> Any:
    if isinstance(body, str):
        try:
            return json.loads(body)
        except ValueError:
            return body
    return body


def _header(headers: dict[str, str], name: str) -> str | None:
    for key, value in headers.items():
        if key.lower() == name.lower():
            return value
    return None


def _body_signals_error(body: Any) -> bool:
    body = _as_json(body)
    if not isinstance(body, dict):
        return False
    return any(body.get(key) for key in ("error", "errors"))  # null / empty = no signal


def _field_named_in_body(body: Any) -> str | None:
    body = _as_json(body)
    if not isinstance(body, dict):
        return None
    candidates = [body]
    for key in ("errors", "details"):
        if isinstance(body.get(key), list):
            candidates += [item for item in body[key] if isinstance(item, dict)]
    for item in candidates:
        for key in FIELD_KEYS:
            if isinstance(item.get(key), str) and item[key]:
                return item[key]
    return None


def _validation(record: LogRecord, spec: dict | None) -> RuleResult:
    op_name = None
    if spec is not None:
        operation = find_operation(spec, record.method, record.url)
        if operation is not None:
            op_name = operation_name(spec, record.method, record.url)
            missing = required_fields_missing(record, operation)
            mismatched = type_mismatches(record, operation)
            if missing or mismatched:
                parts = []
                if missing:
                    parts.append("required field(s) missing from the request: " + ", ".join(missing))
                parts += [f"field '{m.field}' should be {m.expected} but was sent as {m.actual}"
                          for m in mismatched]
                names = missing + [m.field for m in mismatched]
                return _diagnosed(
                    "validation", False, "validation_from_spec",
                    f"The server rejected the request, and the spec shows why: {'; '.join(parts)}.",
                    f"Fix the request body to match the spec for {op_name}: {', '.join(names)}.",
                    SpecReference(op_name, ", ".join(names)),
                )
    named = _field_named_in_body(record.response_body)
    if named:
        return _diagnosed(
            "validation", False, "validation_from_body",
            f"The server rejected the request and its error message points at the field '{named}'. "
            "The spec check found no obvious missing or wrongly typed field.",
            f"Check the value you sent for '{named}' (format, range, allowed values).",
            SpecReference(op_name, named),
        )
    return _needs_llm("400/422 with no spec problem found and no field named in the response")


def apply_rules(record: LogRecord, spec: dict | None, timeout_threshold_ms: float) -> RuleResult:
    result = _apply(record, spec, timeout_threshold_ms)
    # A response with a real status is diagnosed by that status. If it was also slow,
    # say so, but never let slowness change the diagnosis.
    if (result.outcome == "diagnosed" and record.status_code not in (None, 408)
            and record.duration_ms > timeout_threshold_ms):
        result.diagnosis.explanation += (
            f" Note: the call was slow ({record.duration_ms:g} ms, over the "
            f"{timeout_threshold_ms:g} ms threshold).")
    return result


def _apply(record: LogRecord, spec: dict | None, timeout_threshold_ms: float) -> RuleResult:
    status = record.status_code

    # 1. no response at all, or the server itself said 408. Slowness alone does not count.
    if status is None or status == 408:
        if status is None:
            why = "No response was received."
        else:
            why = "The server answered 408 Request Timeout."
        return _diagnosed(
            "network_timeout", True, "network_timeout",
            f"{why} This is usually a network problem or a slow server, not a mistake in the request.",
            "Retry with exponential backoff. If it keeps happening, check connectivity and the server's status.",
        )

    # 2. success codes
    if 200 <= status < 400:
        if _body_signals_error(record.response_body):
            return _needs_llm(f"{status} response but the body contains an error")
        return RuleResult("not_a_failure")

    # 3. rate limit
    if status == 429:
        retry_after = _header(record.response_headers, "Retry-After")
        if retry_after is not None:
            return _diagnosed(
                "rate_limit", True, "rate_limit_with_header",
                f"Too many requests. The server says to wait (Retry-After: {retry_after}).",
                f"Wait the time given in Retry-After ({retry_after}) before retrying, then slow down.",
            )
        return _diagnosed(
            "rate_limit", True, "rate_limit_no_header",
            "Too many requests (429), but the server did not say how long to wait.",
            "Retry with exponential backoff and reduce the request rate.",
        )

    # 4. validation
    if status in (400, 422):
        return _validation(record, spec)

    # 5. auth
    if status == 401:
        return _diagnosed(
            "auth", False, "auth",
            "The server does not accept the credentials (missing, expired or invalid).",
            "Check that the API key or token is present, correct and not expired.",
        )
    if status == 403:
        return _diagnosed(
            "auth", False, "auth",
            "The credentials were understood but do not have permission for this action.",
            "Check the account's permissions or scopes for this endpoint.",
        )

    # 6. not found
    if status == 404:
        return _diagnosed(
            "not_found", False, "not_found",
            "The server has nothing at this URL (wrong path, or the resource ID does not exist).",
            "Check the path and the IDs in it against the spec.",
        )

    # 7. server errors
    if status in (500, 502, 503, 504):
        return _diagnosed(
            "server_error", True, "server_error",
            f"The server failed ({status}). This is usually temporary and not caused by the request.",
            "Retry with exponential backoff. If it persists, contact the API provider.",
        )
    if status == 501:
        return _diagnosed(
            "server_error", False, "server_error",
            "The server does not support this operation (501).",
            "Do not retry. Use a supported endpoint or method.",
        )

    # 8. everything else
    return _needs_llm(f"status {status} has no rule")
