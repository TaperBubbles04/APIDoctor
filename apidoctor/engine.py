"""Run the rules; ask the LLM only when the rules can't decide. Never raises."""
import random
import time
from typing import Callable

from apidoctor.config import Config, load_config
from apidoctor.llm.base import LLMError, LLMProvider, LLMRateLimited
from apidoctor.models import Diagnosis, LogRecord
from apidoctor.rules import apply_rules
from apidoctor.spec import excerpt_for


def _unknown(why: str, detail: str) -> Diagnosis:
    return Diagnosis(
        category="unknown", retryable=False, confidence=0.0,
        explanation=f"No rule could decide ({why}). {detail}",
        suggested_fix="Read the response body and the spec by hand, or try the LLM again later.",
        source="rule",
    )


def ask_llm(record: LogRecord, spec: dict | None, provider: LLMProvider, config: Config,
             sleep: Callable[[float], None]) -> Diagnosis | str:
    """Return the LLM's Diagnosis, or a short reason code if it could not help."""
    excerpt = excerpt_for(record, spec) if spec is not None else None
    for attempt in range(config.max_retries + 1):
        try:
            return provider.diagnose(record, excerpt, None)
        except LLMRateLimited as limited:
            if limited.retry_after is not None and limited.retry_after > config.max_retry_after_s:
                return "retry_after_too_long"  # never sleep longer than the configured cap
            if attempt == config.max_retries:
                return "rate_limited"
            if limited.retry_after is not None:
                wait = limited.retry_after
            else:  # exponential backoff with jitter
                wait = config.delay_between_llm_calls_s * 2 ** attempt * random.uniform(0.5, 1.0)
            sleep(wait)
        except LLMError as error:
            return error.reason
        except Exception:  # a buggy provider must not crash the caller
            return "unexpected"
    return "rate_limited"  # unreachable, keeps the function total


def diagnose(record: LogRecord, spec: dict | None, provider: LLMProvider | None = None,
             config: Config | None = None, sleep: Callable[[float], None] = time.sleep) -> Diagnosis | None:
    """Return a Diagnosis, or None if the call was not a failure."""
    config = config or load_config()
    result = apply_rules(record, spec, config.timeout_threshold_ms)
    if result.outcome == "not_a_failure":
        return None
    if result.outcome == "diagnosed":
        return result.diagnosis
    # needs_llm
    if provider is None:
        return _unknown(result.reason, "The LLM is off, so no deeper analysis was done.")
    answer = ask_llm(record, spec, provider, config, sleep)
    if isinstance(answer, Diagnosis):
        return answer
    return _unknown(result.reason, f"The LLM was unavailable (reason: {answer}), so no deeper analysis was done.")
