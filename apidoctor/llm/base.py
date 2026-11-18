"""The one small interface every LLM backend must follow."""
from typing import Protocol

from apidoctor.config import Config
from apidoctor.models import Diagnosis, LogRecord


class LLMError(Exception):
    """Any LLM failure. The message is only a short reason code, never raw error text."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class LLMRateLimited(LLMError):
    """The provider said 'too many requests'. retry_after is in seconds, or None."""

    def __init__(self, retry_after: float | None = None):
        super().__init__("rate_limited")
        self.retry_after = retry_after


class LLMProvider(Protocol):
    def diagnose(self, record: LogRecord, spec_excerpt: dict | None,
                 rule_hint: Diagnosis | None) -> Diagnosis:
        """Return a Diagnosis with source='llm'. Raise LLMRateLimited or LLMError on failure."""
        ...


def get_provider(config: Config) -> LLMProvider | None:
    """None means the LLM is off (no key): the system then runs on rules only."""
    if not config.llm_enabled:
        return None
    from apidoctor.llm.groq import GroqProvider  # imported here to avoid a circular import
    return GroqProvider(config)
