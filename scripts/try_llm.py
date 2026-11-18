"""Make exactly ONE real LLM call on one sample record. Run by hand, never by tests.

    env -u PYTHONPATH ./venv/bin/python scripts/try_llm.py

Needs GROQ_API_KEY in the environment or .env. Only synthetic, redacted data is sent.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so 'apidoctor' is importable

from apidoctor.config import PROJECT_ROOT, load_config
from apidoctor.llm.base import LLMError, get_provider
from apidoctor.loader import load_logs
from apidoctor.models import LogRecord
from apidoctor.rules import apply_rules
from apidoctor.spec import excerpt_for, load_spec


def main() -> int:
    config = load_config()
    provider = get_provider(config)
    if provider is None:
        print("No GROQ_API_KEY found (environment or .env). LLM is off; nothing was sent.")
        return 1
    spec = load_spec(PROJECT_ROOT / "sample_data" / "spec.yaml")
    records, _ = load_logs(PROJECT_ROOT / "sample_data" / "logs.jsonl", config)
    record = next(
        (r for r in records if apply_rules(r, spec, config.timeout_threshold_ms).outcome == "needs_llm"),
        None,
    )
    if record is None:  # none of the sample logs is ambiguous, so use one synthetic 409
        record = LogRecord(
            timestamp="2026-01-01T11:00:00Z", method="POST", url="https://api.example.test/v1/orders",
            status_code=409, request_headers={"Content-Type": "application/json"},
            request_body={"customer": {"name": "Test User"}, "quantity": 2, "note": "duplicate?"},
            response_body={"error": "conflict", "detail": "an order with this note already exists"},
            duration_ms=60,
        )
    print(f"Sending ONE call for: {record.method} {record.url} -> {record.status_code} (model: {config.llm_model})")
    try:
        diagnosis = provider.diagnose(record, excerpt_for(record, spec), None)
    except LLMError as error:
        print(f"LLM call failed (reason code: {error.reason})")
        return 2
    print(json.dumps(diagnosis.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
