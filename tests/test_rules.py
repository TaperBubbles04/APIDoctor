import unittest
from pathlib import Path

from apidoctor.config import load_config
from apidoctor.engine import diagnose
from apidoctor.loader import load_logs
from apidoctor.models import LogRecord
from apidoctor.rules import CONFIDENCE, apply_rules
from apidoctor.spec import load_spec

ROOT = Path(__file__).resolve().parent.parent
SPEC = load_spec(ROOT / "sample_data" / "spec.yaml")
CONFIG = load_config(env_path=Path("/nonexistent/.env"))
LIMIT = 10000


def rec(status=200, method="GET", url="https://api.example.test/v1/items",
        body=None, resp=None, resp_headers=None, duration=100):
    return LogRecord("t", method, url, status, {}, body, resp, duration, resp_headers or {})


def run(record, spec=SPEC):
    return apply_rules(record, spec, LIMIT)


def diag(record, spec=SPEC):
    result = run(record, spec)
    assert result.outcome == "diagnosed", result
    return result.diagnosis


class Rule1Timeout(unittest.TestCase):
    def test_no_status(self):
        d = diag(rec(status=None))
        self.assertEqual((d.category, d.retryable, d.source), ("network_timeout", True, "rule"))

    def test_slow_response_is_diagnosed_by_its_status_with_a_slow_note(self):
        for status, category in ((401, "auth"), (422, "validation"), (500, "server_error"), (404, "not_found")):
            d = diag(rec(status=status, duration=LIMIT + 1, resp={"field": "x"}))
            self.assertEqual(d.category, category)
            self.assertIn("was slow", d.explanation)

    def test_fast_response_has_no_slow_note(self):
        self.assertNotIn("slow", diag(rec(status=500, duration=5)).explanation)

    def test_no_status_stays_a_timeout_without_extra_slow_note(self):
        d = diag(rec(status=None, duration=LIMIT * 5))
        self.assertEqual(d.category, "network_timeout")
        self.assertNotIn("was slow", d.explanation)

    def test_slow_success_is_not_a_timeout(self):
        for status in (200, 204, 301):
            self.assertEqual(run(rec(status=status, duration=LIMIT * 5)).outcome, "not_a_failure")

    def test_slow_success_with_error_body_needs_llm(self):
        result = run(rec(status=200, duration=LIMIT * 5, resp={"error": "partial failure"}))
        self.assertEqual(result.outcome, "needs_llm")

    def test_exactly_at_limit_is_not_slow(self):
        self.assertEqual(diag(rec(status=500, duration=LIMIT)).category, "server_error")

    def test_408_is_a_timeout_even_if_fast(self):
        self.assertEqual(diag(rec(status=408, duration=5)).category, "network_timeout")

    def test_408(self):
        d = diag(rec(status=408))
        self.assertEqual((d.category, d.retryable), ("network_timeout", True))


class Rule2Success(unittest.TestCase):
    def test_200_and_3xx_not_a_failure(self):
        for status in (200, 201, 204, 301, 304):
            self.assertEqual(run(rec(status=status)).outcome, "not_a_failure")

    def test_200_with_error_body_needs_llm(self):
        self.assertEqual(run(rec(resp={"error": "quota exceeded"})).outcome, "needs_llm")
        self.assertEqual(run(rec(resp={"errors": [{"m": "x"}]})).outcome, "needs_llm")
        self.assertEqual(run(rec(resp='{"error": "as json text"}')).outcome, "needs_llm")

    def test_200_with_null_or_empty_error_is_fine(self):
        self.assertEqual(run(rec(resp={"error": None, "data": 1})).outcome, "not_a_failure")
        self.assertEqual(run(rec(resp={"errors": []})).outcome, "not_a_failure")


class Rule3RateLimit(unittest.TestCase):
    def test_retry_after_any_letter_case(self):
        for name in ("Retry-After", "retry-after", "RETRY-AFTER"):
            d = diag(rec(status=429, resp_headers={name: "30"}))
            self.assertEqual((d.category, d.retryable), ("rate_limit", True))
            self.assertEqual(d.confidence, CONFIDENCE["rate_limit_with_header"])
            self.assertIn("30", d.explanation)

    def test_without_retry_after_is_less_confident(self):
        d = diag(rec(status=429))
        self.assertEqual((d.category, d.retryable), ("rate_limit", True))
        self.assertEqual(d.confidence, CONFIDENCE["rate_limit_no_header"])
        self.assertLess(CONFIDENCE["rate_limit_no_header"], CONFIDENCE["rate_limit_with_header"])


class Rule4Validation(unittest.TestCase):
    URL = "https://api.example.test/v1/orders"

    def test_missing_required_field_from_spec(self):
        d = diag(rec(status=422, method="POST", url=self.URL, body={"customer": {"name": "x"}}))
        self.assertEqual((d.category, d.retryable), ("validation", False))
        self.assertEqual(d.spec_reference.operation, "createOrder")
        self.assertEqual(d.spec_reference.field, "quantity")
        self.assertEqual(d.confidence, CONFIDENCE["validation_from_spec"])

    def test_type_mismatch_from_spec(self):
        d = diag(rec(status=400, method="POST", url=self.URL, body={"customer": {}, "quantity": "three"}))
        self.assertEqual(d.category, "validation")
        self.assertEqual(d.spec_reference.field, "quantity")
        self.assertIn("integer", d.explanation)

    def test_field_named_in_response_body(self):
        d = diag(rec(status=422, method="POST", url=self.URL, body={"customer": {}, "quantity": 2},
                     resp={"error": "bad", "field": "customer.email"}))
        self.assertEqual(d.category, "validation")
        self.assertEqual(d.spec_reference.field, "customer.email")
        self.assertEqual(d.confidence, CONFIDENCE["validation_from_body"])
        self.assertLess(CONFIDENCE["validation_from_body"], CONFIDENCE["validation_from_spec"])

    def test_field_named_inside_errors_list_without_spec(self):
        d = diag(rec(status=400, resp={"errors": [{"param": "limit"}]}), spec=None)
        self.assertEqual(d.spec_reference.field, "limit")

    def test_nothing_to_go_on_needs_llm(self):
        result = run(rec(status=400, method="POST", url=self.URL, body={"customer": {}, "quantity": 2},
                         resp={"error": "bad request"}))
        self.assertEqual(result.outcome, "needs_llm")


class Rules5to8StatusCodes(unittest.TestCase):
    def test_401(self):
        d = diag(rec(status=401))
        self.assertEqual((d.category, d.retryable), ("auth", False))

    def test_403(self):
        d = diag(rec(status=403))
        self.assertEqual((d.category, d.retryable), ("auth", False))

    def test_404(self):
        d = diag(rec(status=404))
        self.assertEqual((d.category, d.retryable), ("not_found", False))

    def test_retryable_5xx(self):
        for status in (500, 502, 503, 504):
            d = diag(rec(status=status))
            self.assertEqual((d.category, d.retryable), ("server_error", True))

    def test_501_not_retryable(self):
        d = diag(rec(status=501))
        self.assertEqual((d.category, d.retryable), ("server_error", False))

    def test_everything_else_needs_llm(self):
        for status in (409, 418, 507, 100):
            self.assertEqual(run(rec(status=status)).outcome, "needs_llm")

    def test_every_diagnosis_has_explanation_and_fix(self):
        for status in (None, 408, 429, 401, 403, 404, 500, 501):
            d = diag(rec(status=status))
            self.assertTrue(d.explanation.strip() and d.suggested_fix.strip())


class EngineTests(unittest.TestCase):
    def test_llm_off_gives_unknown(self):
        d = diagnose(rec(status=409), SPEC, config=CONFIG)
        self.assertEqual((d.category, d.source, d.retryable), ("unknown", "rule", False))
        self.assertIn("LLM is off", d.explanation)

    def test_not_a_failure_returns_none(self):
        self.assertIsNone(diagnose(rec(status=200), SPEC, config=CONFIG))

    def test_diagnosed_passes_through(self):
        self.assertEqual(diagnose(rec(status=404), SPEC, config=CONFIG).category, "not_found")

    def test_broken_provider_does_not_crash_the_engine(self):
        d = diagnose(rec(status=409), SPEC, provider=object(), config=CONFIG)
        self.assertEqual(d.category, "unknown")
        self.assertIn("unexpected", d.explanation)


class SampleLogsEndToEnd(unittest.TestCase):
    def test_all_six_sample_records(self):
        records, errors = load_logs(ROOT / "sample_data" / "logs.jsonl", CONFIG)
        self.assertEqual(errors, [])
        out = []
        for r in records:
            d = diagnose(r, SPEC, config=CONFIG)
            out.append(None if d is None else (d.category, d.retryable))
        self.assertEqual(out, [
            None,                         # 200
            ("auth", False),              # 401
            ("validation", False),        # 422: quantity "three"
            ("rate_limit", True),         # 429 (sample has no Retry-After header)
            ("server_error", True),       # 500
            ("network_timeout", True),    # no status
        ])
        self.assertEqual(diagnose(records[2], SPEC, config=CONFIG).spec_reference.field, "quantity")


if __name__ == "__main__":
    unittest.main()
