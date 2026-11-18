"""A provider wrapper that adds a cache, a call limit and a pause between real calls."""
import hashlib
import json
import time
from pathlib import Path
from typing import Callable

from apidoctor.config import Config
from apidoctor.llm.base import LLMError
from apidoctor.llm.prompt import PROMPT_VERSION, redacted_record
from apidoctor.models import Diagnosis, LogRecord, SpecReference


def diagnosis_from_dict(d: dict) -> Diagnosis:
    return Diagnosis(
        category=d["category"], retryable=d["retryable"], confidence=d["confidence"],
        explanation=d["explanation"], suggested_fix=d["suggested_fix"], source=d["source"],
        spec_reference=SpecReference(**d["spec_reference"]),
    )


class LimitedCachedProvider:
    """Wraps a real provider: cache first, then the call limit, then a pause, then the call."""

    def __init__(self, inner, config: Config, cache_path: Path, max_calls: int,
                 sleep: Callable[[float], None] = time.sleep):
        self.inner, self.config, self.max_calls, self.sleep = inner, config, max_calls, sleep
        self.cache_path = Path(cache_path)
        self.real_calls = 0
        self.cache_hits = 0
        self.refused = False   # set when a call was refused for quota; the caller resets it per record
        self.cache: dict = {}
        if self.cache_path.is_file():
            try:
                self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
                if not isinstance(self.cache, dict):
                    raise ValueError
            except ValueError:
                raise ValueError(f"cache file {self.cache_path.name} is not valid; move or delete it") from None

    def key(self, record: LogRecord, spec_excerpt: dict | None) -> str:
        material = [self.config.llm_model, PROMPT_VERSION,
                    redacted_record(record, self.config.redaction_min_token_length), spec_excerpt]
        text = json.dumps(material, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def diagnose(self, record, spec_excerpt, rule_hint) -> Diagnosis:
        key = self.key(record, spec_excerpt)
        if key in self.cache:
            try:
                found = diagnosis_from_dict(self.cache[key])
                self.cache_hits += 1
                return found
            except (KeyError, TypeError, ValueError):
                pass   # unusable entry: treat as a miss
        if self.real_calls >= self.max_calls:
            self.refused = True
            raise LLMError("quota_skipped")
        if self.real_calls > 0:
            self.sleep(self.config.delay_between_llm_calls_s)
        self.real_calls += 1
        answer = self.inner.diagnose(record, spec_excerpt, rule_hint)   # failures are not cached
        self.cache[key] = answer.to_dict()
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, indent=1), encoding="utf-8")
        return answer
