"""Groq backend (OpenAI-compatible chat endpoint). Standard library only."""
import json
import urllib.error
import urllib.request
from typing import Callable

from apidoctor.config import Config
from apidoctor.llm.base import LLMError, LLMRateLimited
from apidoctor.llm.prompt import DIAGNOSIS_SCHEMA, build_messages
from apidoctor.models import Diagnosis, LogRecord, SpecReference

USER_AGENT = "apidoctor/0.1"

# transport(url, headers, payload, timeout_s) -> (status_code, response_headers, body_text)
Transport = Callable[[str, dict, dict, float], tuple[int, dict, str]]


def urllib_transport(url: str, headers: dict, payload: dict, timeout: float) -> tuple[int, dict, str]:
    """The real HTTP call. Error bodies are dropped so they cannot leak into messages."""
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, dict(response.headers), response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:   # must come before URLError (it is a subclass)
        return e.code, dict(e.headers or {}), ""
    except urllib.error.URLError as e:
        raise LLMError("timeout" if isinstance(e.reason, TimeoutError) else "network") from None
    except TimeoutError:
        raise LLMError("timeout") from None
    except OSError:
        raise LLMError("network") from None


def _retry_after(headers: dict) -> float | None:
    for name, value in headers.items():
        if name.lower() == "retry-after":
            try:
                seconds = float(value)
            except (TypeError, ValueError):
                return None   # e.g. an HTTP date: not handled, caller uses backoff
            return seconds if seconds >= 0 else None
    return None


def parse_diagnosis(text: str) -> Diagnosis:
    """Turn the model's reply into a Diagnosis, or raise ValueError with a short reason."""
    try:
        obj = json.loads(text)
    except ValueError:
        raise ValueError("reply is not a single valid JSON object") from None
    if not isinstance(obj, dict):
        raise ValueError("reply is not a JSON object")
    try:
        category, retryable, confidence = obj["category"], obj["retryable"], obj["confidence"]
        explanation, fix = obj["explanation"], obj["suggested_fix"]
    except KeyError as missing:
        raise ValueError(f"missing field {missing}") from None
    if not isinstance(retryable, bool):
        raise ValueError("'retryable' must be true or false")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("'confidence' must be a number")
    if not isinstance(explanation, str) or not isinstance(fix, str):
        raise ValueError("'explanation' and 'suggested_fix' must be strings")
    ref = obj.get("spec_reference") or {}
    if not isinstance(ref, dict):
        raise ValueError("'spec_reference' must be an object")
    operation, field = ref.get("operation"), ref.get("field")
    if not all(v is None or isinstance(v, str) for v in (operation, field)):
        raise ValueError("spec_reference parts must be strings or null")
    return Diagnosis(  # raises ValueError for a bad category or confidence outside 0-1
        category=category, retryable=retryable, confidence=float(confidence),
        explanation=explanation, suggested_fix=fix, source="llm",
        spec_reference=SpecReference(operation, field),
    )


class GroqProvider:
    def __init__(self, config: Config, transport: Transport = urllib_transport):
        self._config = config
        self._transport = transport

    def __repr__(self) -> str:  # never shows the key
        return f"GroqProvider(model={self._config.llm_model!r})"

    def diagnose(self, record: LogRecord, spec_excerpt: dict | None,
                 rule_hint: Diagnosis | None) -> Diagnosis:
        messages = build_messages(record, spec_excerpt, rule_hint,
                                  self._config.redaction_min_token_length,
                                  self._config.max_body_chars)
        reply = self._chat(messages)
        try:
            return parse_diagnosis(reply)
        except ValueError as problem:
            repair = messages + [
                {"role": "assistant", "content": reply},
                {"role": "user", "content": (
                    f"Your reply was invalid: {str(problem)[:200]}. "
                    "Reply again with ONE JSON object only, matching the schema.")},
            ]
        try:
            return parse_diagnosis(self._chat(repair))
        except ValueError:
            raise LLMError("bad_reply") from None

    def _chat(self, messages: list[dict]) -> str:
        cfg = self._config
        payload = {
            "model": cfg.llm_model,
            "messages": messages,
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "diagnosis", "strict": False, "schema": DIAGNOSIS_SCHEMA},
            },
        }
        headers = {
            "Authorization": f"Bearer {cfg.api_key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }
        url = cfg.llm_base_url.rstrip("/") + "/chat/completions"
        status, response_headers, body = self._transport(url, headers, payload, cfg.llm_timeout_s)
        if status == 429:
            raise LLMRateLimited(_retry_after(response_headers))
        if status != 200:
            raise LLMError(f"http_{status}")
        try:
            content = json.loads(body)["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError):
            raise LLMError("bad_response") from None
        if not isinstance(content, str):
            raise LLMError("bad_response")
        return content
