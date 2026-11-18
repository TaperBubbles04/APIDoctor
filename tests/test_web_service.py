import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from apidoctor.config import load_config
from apidoctor.loader import load_logs_text
from apidoctor.models import Diagnosis
from apidoctor.service import (
    MAX_BODY_BYTES, MAX_LOG_LINES, RequestError, build_report, diagnose_request, diagnose_logs, parse_payload,
)

ROOT = Path(__file__).resolve().parent.parent
SPEC_TEXT = (ROOT / "sample_data" / "spec.yaml").read_text(encoding="utf-8")
LOGS_TEXT = (ROOT / "sample_data" / "logs.jsonl").read_text(encoding="utf-8")
CONFIG = replace(load_config(env_path=Path("/nonexistent/.env")), api_key=None,
                 delay_between_llm_calls_s=0, max_llm_calls_per_run=1)


class Scripted:
    def __init__(self):
        self.calls = 0

    def diagnose(self, record, spec_excerpt, rule_hint):
        self.calls += 1
        return Diagnosis("conflict", False, 0.8, "scripted", "scripted fix", "llm")


def line(i, status=409):
    return json.dumps({"timestamp": "t", "method": "PUT", "url": f"https://api.example.test/v1/orders/ord_{i}",
                       "status_code": status, "request_headers": {}, "response_body": {"error": "x"},
                       "duration_ms": 5})


def payload(logs=LOGS_TEXT, spec=SPEC_TEXT, use_llm=False):
    return {"logs_text": logs, "spec_text": spec, "use_llm": use_llm}


def raw(**fields):
    return json.dumps(fields).encode()


class ParsePayloadTests(unittest.TestCase):
    def test_valid(self):
        p = parse_payload(raw(logs_text="x", spec_text="y"))
        self.assertEqual(p, {"logs_text": "x", "spec_text": "y", "use_llm": False})

    def test_rejections_use_fixed_messages(self):
        cases = [
            (b"{nope", 400), (b"[1]", 400),
            (raw(logs_text=1, spec_text="y"), 400), (raw(logs_text="x"), 400),
            (raw(logs_text="x", spec_text="y", use_llm="yes"), 400),
            (raw(logs_text="   ", spec_text="y"), 400),
        ]
        for body, status in cases:
            with self.assertRaises(RequestError) as ctx:
                parse_payload(body)
            self.assertEqual(ctx.exception.status, status, body)

    def test_size_and_line_limits(self):
        with self.assertRaises(RequestError) as ctx:
            parse_payload(b" " * (MAX_BODY_BYTES + 1))
        self.assertEqual(ctx.exception.status, 413)
        many = "\n".join(line(i) for i in range(MAX_LOG_LINES + 1))
        with self.assertRaises(RequestError) as ctx:
            parse_payload(raw(logs_text=many, spec_text="y"))
        self.assertEqual(ctx.exception.status, 413)
        ok = "\n".join(line(i) for i in range(MAX_LOG_LINES))
        self.assertEqual(len(parse_payload(raw(logs_text=ok, spec_text="y"))["logs_text"].splitlines()), MAX_LOG_LINES)


class DiagnoseRequestTests(unittest.TestCase):
    def run_request(self, p, provider=None):
        with tempfile.TemporaryDirectory() as d:
            return diagnose_request(p, CONFIG, provider, Path(d) / "cache.json", sleep=lambda s: None)

    def test_sample_data_rules_only(self):
        report = self.run_request(payload())
        self.assertEqual((report["total_records"], report["failures"], report["non_failures"]), (6, 5, 1))
        self.assertEqual(len(report["groups"]), 5)
        self.assertIsNone(report["llm"])
        self.assertEqual(report["load_errors"], [])
        self.assertNotIn("SYNTHETIC", json.dumps(report))

    def test_load_errors_are_reported(self):
        report = self.run_request(payload(logs=line(1, 404) + "\n{broken\n"))
        self.assertEqual(report["load_errors"][0]["line"], 2)
        self.assertEqual(report["failures"], 1)

    def test_bad_spec_is_a_clean_400(self):
        with self.assertRaises(RequestError) as ctx:
            self.run_request(payload(spec="a: [unclosed SYNTHETIC-SECRET"))
        self.assertEqual(ctx.exception.status, 400)
        self.assertNotIn("SYNTHETIC", ctx.exception.message)

    def test_use_llm_without_key_is_rejected(self):
        with self.assertRaises(RequestError) as ctx:
            self.run_request(payload(use_llm=True), provider=None)
        self.assertEqual(ctx.exception.status, 400)
        self.assertIn("no LLM key", ctx.exception.message)

    def test_llm_call_cap_per_request_comes_from_config(self):
        fake = Scripted()
        logs = "\n".join(line(i) for i in (1, 2, 3))
        report = self.run_request(payload(logs=logs, use_llm=True), provider=fake)
        self.assertEqual(fake.calls, CONFIG.max_llm_calls_per_run)   # cap is 1
        self.assertEqual((report["llm"]["real_calls"], report["llm_skipped"]), (1, 2))

    def test_llm_off_means_no_provider_calls(self):
        fake = Scripted()
        self.run_request(payload(logs=line(1), use_llm=False), provider=fake)
        self.assertEqual(fake.calls, 0)

    def test_corrupt_cache_is_a_clean_500(self):
        with tempfile.TemporaryDirectory() as d:
            cache = Path(d) / "cache.json"
            cache.write_text("{broken")
            with self.assertRaises(RequestError) as ctx:
                diagnose_request(payload(logs=line(1), use_llm=True), CONFIG, Scripted(), cache)
            self.assertEqual(ctx.exception.status, 500)


class LoadLogsTextTests(unittest.TestCase):
    def test_same_as_file_loading(self):
        records, errors = load_logs_text(LOGS_TEXT + "\n\nnot json\n", CONFIG)
        self.assertEqual((len(records), [e.line_number for e in errors]), (6, [9]))   # lines 7-8 are blank

    def test_build_report_shape(self):
        records, errors = load_logs_text(line(1, 404), CONFIG)
        report = build_report(diagnose_logs(records, None, None, CONFIG), errors, None)
        self.assertEqual(set(report), {"total_records", "failures", "non_failures", "llm_skipped",
                                       "groups", "results", "load_errors", "llm"})


if __name__ == "__main__":
    unittest.main()
