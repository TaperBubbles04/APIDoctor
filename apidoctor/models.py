"""Data shapes: one log record in, one diagnosis out."""
from dataclasses import dataclass, field, asdict
from typing import Any

CATEGORIES = (
    "auth", "validation", "not_found", "conflict", "rate_limit",
    "server_error", "network_timeout", "unknown",
)
SOURCES = ("rule", "llm")


@dataclass
class LogRecord:
    timestamp: str
    method: str
    url: str
    status_code: int | None          # None = no response (network failure)
    request_headers: dict[str, str]
    request_body: Any                # any JSON value, or None
    response_body: Any
    duration_ms: float
    response_headers: dict[str, str] = field(default_factory=dict)


@dataclass
class SpecReference:
    operation: str | None = None
    field: str | None = None


@dataclass
class Diagnosis:
    category: str
    retryable: bool
    confidence: float
    explanation: str
    suggested_fix: str
    source: str
    spec_reference: SpecReference = field(default_factory=SpecReference)

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES:
            raise ValueError(f"bad category: {self.category!r}")
        if self.source not in SOURCES:
            raise ValueError(f"bad source: {self.source!r}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence out of range: {self.confidence!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
