"""Load settings from config.json and the API key from the environment / .env."""
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KEY_NAME = "GROQ_API_KEY"


@dataclass(frozen=True)
class Config:
    llm_model: str
    llm_base_url: str
    max_retries: int
    llm_timeout_s: float
    max_retry_after_s: float
    max_body_chars: int
    max_llm_calls_per_run: int
    delay_between_llm_calls_s: float
    redaction_min_token_length: int
    timeout_threshold_ms: float
    # repr=False: printing a Config must never show the key.
    api_key: str | None = field(default=None, repr=False)

    @property
    def llm_enabled(self) -> bool:
        return bool(self.api_key)


def read_env_file(path: Path) -> dict[str, str]:
    """Parse simple KEY=VALUE lines. Ignores blanks and # comments."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        values[name.strip()] = value
    return values


def load_config(config_path: Path | None = None, env_path: Path | None = None) -> Config:
    config_path = config_path or PROJECT_ROOT / "config.json"
    env_path = env_path or PROJECT_ROOT / ".env"
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    # Real environment variable wins over .env. Empty string counts as "no key".
    api_key = os.environ.get(KEY_NAME) or read_env_file(env_path).get(KEY_NAME) or None
    try:
        return Config(
            llm_model=str(raw["llm_model"]),
            llm_base_url=str(raw["llm_base_url"]),
            max_retries=int(raw["max_retries"]),
            llm_timeout_s=float(raw["llm_timeout_s"]),
            max_retry_after_s=float(raw["max_retry_after_s"]),
            max_body_chars=int(raw["max_body_chars"]),
            max_llm_calls_per_run=int(raw["max_llm_calls_per_run"]),
            delay_between_llm_calls_s=float(raw["delay_between_llm_calls_s"]),
            redaction_min_token_length=int(raw["redaction_min_token_length"]),
            timeout_threshold_ms=float(raw["timeout_threshold_ms"]),
            api_key=api_key,
        )
    except KeyError as missing:
        raise ValueError(f"config.json is missing setting {missing}") from None
