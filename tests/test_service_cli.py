import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from apidoctor import cli
from apidoctor.config import load_config
from apidoctor.models import Diagnosis, LogRecord
from apidoctor.service import UNKNOWN_OPERATION, diagnose_logs
from apidoctor.spec import load_spec, load_spec_text

ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = ROOT / "sample_data" / "spec.yaml"
LOGS_PATH = ROOT / "sample_data" / "logs.jsonl"
SPEC = load_spec(SPEC_PATH)
NO_KEY = replace(load_config(env_path=Path("/nonexistent/.env")), api_key=None, delay_between_llm_calls_s=0)


def rec(status=409, method="PUT", url="https://api.example.test/v1/orders/ord_1", body=None, resp=None):
    return LogRecord("t", method, url, status, {}, body, resp or {"error": "x"}, 10)


class Scripted:
    """Fake provider; retryable alternates when alternate=True."""
    def __init__(self, alternate=False):
        self.calls, self.alternate = 0, alternate

    def diagnose(self, record, spec_excerpt, rule_hint):
        self.calls += 1
        retry = bool(self.calls % 2) if self.alternate else False
        return Diagnosis("conflict", retry, 0.8, "scripted", "scripted fix", "llm")


class ServiceTests(unittest.TestCase):
    def test_grouping_counts_and_sorting(self):
        bad_body = {"customer": {"name": "x"}, "quantity": "three"}
        records = [rec(429, "GET", "https://api.example.test/v1/items"),
                   rec(422, "POST", "https://api.example.test/v1/orders", bad_body),
                   rec(422, "POST", "https://api.example.test/v1/orders", bad_body),
                   rec(422, "POST", "https://api.example.test/v1/orders", bad_body),
                   rec(429, "GET", "https://api.example.test/v1/items")]
        out = diagnose_logs(records, SPEC, None, NO_KEY)
        self.assertEqual([(g.operation, g.category, g.count, g.retryable) for g in out.groups],
                         [("createOrder", "validation", 3, False), ("listItems", "rate_limit", 2, True)])
        self.assertEqual((out.total_records, len(out.results)), (5, 5))

    def test_non_failures_are_skipped(self):
        out = diagnose_logs([rec(200, "GET", "https://api.example.test/v1/items", resp={"ok": 1}),
                             rec(404, "GET", "https://api.example.test/v1/orders/ord_9")], SPEC, None, NO_KEY)
        self.assertEqual((out.total_records, out.non_failures, len(out.results)), (2, 1, 1))

    def test_unknown_operation_and_no_spec(self):
        r = rec(404, "GET", "https://api.example.test/v9/nothing")
        self.assertEqual(diagnose_logs([r], SPEC, None, NO_KEY).groups[0].operation, UNKNOWN_OPERATION)
        self.assertEqual(diagnose_logs([r], None, None, NO_KEY).groups[0].operation, UNKNOWN_OPERATION)

    def test_mixed_retryable_flag(self):
        out = diagnose_logs([rec(), rec(url="https://api.example.test/v1/orders/ord_2")], SPEC,
                            Scripted(alternate=True), NO_KEY, sleep=lambda s: None)
        self.assertEqual((out.groups[0].count, out.groups[0].retryable), (2, None))

    def test_to_dict_is_json_safe(self):
        text = json.dumps(diagnose_logs([rec(404, "GET", "https://api.example.test/v1/orders/o")], SPEC, None,
                                        NO_KEY).to_dict())
        self.assertIn('"failures": 1', text)


class SpecTextTests(unittest.TestCase):
    def test_json_and_yaml_text(self):
        self.assertEqual(load_spec_text('{"a": 1}'), {"a": 1})
        self.assertEqual(load_spec_text("a: 1\nb: [2]"), {"a": 1, "b": [2]})

    def test_bad_text_raises_value_error(self):
        for bad in ("a: [unclosed", "just a string", "- 1\n- 2", ""):
            with self.assertRaises(ValueError, msg=bad):
                load_spec_text(bad)


def run_cli(*argv, config=NO_KEY, provider=None):
    out, err = io.StringIO(), io.StringIO()
    with tempfile.TemporaryDirectory() as d, \
            mock.patch("apidoctor.cli.load_config", return_value=config), \
            mock.patch("apidoctor.cli.CACHE_PATH", Path(d) / "cache.json"), \
            mock.patch("apidoctor.cli.get_provider", return_value=provider), \
            contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(list(argv), sleep=lambda s: None)
    return code, out.getvalue(), err.getvalue()


def write(tmp, name, text):
    path = Path(tmp) / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def jsonl(*records):
    return "\n".join(json.dumps(r) for r in records) + "\n"


def raw(status=409, **over):
    r = {"timestamp": "t", "method": "PUT", "url": "https://api.example.test/v1/orders/ord_1",
         "status_code": status, "request_headers": {}, "request_body": None,
         "response_body": {"error": "x"}, "duration_ms": 10}
    r.update(over)
    return r


class CliTests(unittest.TestCase):
    def test_sample_data_rules_only(self):
        code, out, _ = run_cli("diagnose", str(LOGS_PATH), "--spec", str(SPEC_PATH))
        self.assertEqual(code, 0)
        self.assertIn("Failures", out)
        self.assertIn("Summary", out)
        self.assertIn("6 records, 5 failures, 1 not failures (skipped), 0 load errors", out)
        for needle in ("createOrder  validation", "listOrders  auth", "listItems  rate_limit",
                       "updateOrder  server_error", "getReports  network_timeout"):
            self.assertIn(needle, out)
        self.assertNotIn("LLM:", out)           # LLM off by default
        self.assertNotIn("SYNTHETIC", out)      # redacted data only

    def test_llm_is_off_by_default(self):
        with tempfile.TemporaryDirectory() as d:
            code, out, _ = run_cli("diagnose", write(d, "l.jsonl", jsonl(raw())), "--spec", str(SPEC_PATH),
                                   provider=Scripted())
        self.assertIn("The LLM is off", out)
        self.assertIn("unknown", out)

    def test_json_output(self):
        code, out, _ = run_cli("diagnose", str(LOGS_PATH), "--spec", str(SPEC_PATH), "--json")
        data = json.loads(out)
        self.assertEqual((code, data["total_records"], data["failures"], data["non_failures"]), (0, 6, 5, 1))
        self.assertEqual(data["load_errors"], [])
        self.assertIsNone(data["llm"])
        self.assertEqual(sum(g["count"] for g in data["groups"]), 5)
        self.assertEqual({"operation", "category", "count", "retryable"}, set(data["groups"][0]))
        self.assertNotIn("SYNTHETIC", out)

    def test_load_error_is_reported_with_line_number(self):
        with tempfile.TemporaryDirectory() as d:
            logs = write(d, "l.jsonl", json.dumps(raw(404)) + "\n{broken SYNTHETIC-TOKEN-0123456789abcdef\n")
            code, out, _ = run_cli("diagnose", logs, "--spec", str(SPEC_PATH))
            self.assertEqual(code, 0)
            self.assertIn("load error: line 2:", out)
            self.assertNotIn("SYNTHETIC", out)
            code, out, _ = run_cli("diagnose", logs, "--spec", str(SPEC_PATH), "--json")
            self.assertEqual(json.loads(out)["load_errors"][0]["line"], 2)

    def test_llm_without_key_exits_nonzero_with_clear_message(self):
        code, out, err = run_cli("diagnose", str(LOGS_PATH), "--spec", str(SPEC_PATH), "--llm", provider=None)
        self.assertNotEqual(code, 0)
        self.assertIn("GROQ_API_KEY", err)
        self.assertEqual(out, "")

    def test_llm_with_fake_provider_and_call_limit(self):
        records = [raw(url=f"https://api.example.test/v1/orders/ord_{i}") for i in (1, 2, 3)]
        with tempfile.TemporaryDirectory() as d:
            fake = Scripted()
            logs = write(d, "l.jsonl", jsonl(*records))
            code, out, _ = run_cli("diagnose", logs, "--spec", str(SPEC_PATH), "--llm",
                                   "--max-llm-calls", "2", provider=fake)
        self.assertEqual((code, fake.calls), (0, 2))
        self.assertIn("LLM: 2 real calls, 0 cache hits, 1 records skipped (call limit)", out)
        self.assertIn("conflict", out)

    def test_bad_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            good_logs = write(d, "l.jsonl", jsonl(raw(404)))
            self.assertEqual(run_cli("diagnose", f"{d}/missing.jsonl", "--spec", str(SPEC_PATH))[0], 2)
            self.assertEqual(run_cli("diagnose", good_logs, "--spec", f"{d}/missing.yaml")[0], 2)
            self.assertEqual(run_cli("diagnose", good_logs, "--spec", write(d, "s.yaml", "a: [unclosed"))[0], 2)
            self.assertEqual(run_cli("diagnose", good_logs, "--spec", write(d, "s2.yaml", "- 1\n- 2"))[0], 2)
            self.assertEqual(run_cli("diagnose", write(d, "bad.jsonl", "{nope\n"), "--spec", str(SPEC_PATH))[0], 1)


if __name__ == "__main__":
    unittest.main()
