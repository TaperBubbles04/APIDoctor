import json
import os
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

from apidoctor.config import load_config
from apidoctor.loader import load_logs
from apidoctor.models import Diagnosis

ROOT = Path(__file__).resolve().parent.parent
CONFIG = load_config(env_path=Path("/nonexistent/.env"))


def good(**over):
    rec = {"timestamp": "t", "method": "GET", "url": "https://h.test/x",
           "status_code": 200, "request_headers": {}, "request_body": None,
           "response_body": None, "duration_ms": 1}
    rec.update(over)
    return json.dumps(rec)


def load_text(text):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "logs.jsonl"
        p.write_text(text, encoding="utf-8")
        return load_logs(p, CONFIG)


class LoaderTests(unittest.TestCase):
    def test_valid_lines_and_blank_lines(self):
        records, errors = load_text(good() + "\n\n" + good(status_code=None) + "\n")
        self.assertEqual(len(records), 2)
        self.assertEqual(errors, [])
        self.assertIsNone(records[1].status_code)

    def test_malformed_line_reported_with_line_number_and_others_load(self):
        records, errors = load_text(good() + "\n{not json SYNTHETIC-TOKEN-0123456789abcdef\n" + good() + "\n")
        self.assertEqual(len(records), 2)
        self.assertEqual([e.line_number for e in errors], [2])
        self.assertNotIn("SYNTHETIC", errors[0].message)  # error text must not echo content

    def test_missing_and_wrong_type_fields(self):
        bad_missing = json.loads(good()); del bad_missing["url"]
        text = "\n".join([
            json.dumps(bad_missing),
            good(status_code="200"),
            good(duration_ms=-5),
            good(request_headers={"a": 1}),
            "[1, 2]",
        ])
        records, errors = load_text(text)
        self.assertEqual(records, [])
        self.assertEqual([e.line_number for e in errors], [1, 2, 3, 4, 5])

    def test_redacts_before_returning(self):
        tok = "SYNTHETIC-TOKEN-0123456789abcdef"
        records, _ = load_text(good(
            url=f"https://h.test/x?api_key={tok}",
            request_headers={"Authorization": f"Bearer {tok}"},
            request_body={"deep": {"list": [{"v": tok}]}},
            response_body=f"oops {tok}",
        ))
        self.assertNotIn("SYNTHETIC", json.dumps(asdict(records[0])))

    def test_response_headers_loaded_defaulted_validated_and_redacted(self):
        records, errors = load_text("\n".join([
            good(response_headers={"Retry-After": "30", "Set-Cookie": "SYNTHETIC-COOKIE-0123456789abc"}),
            good(),
            good(response_headers={"a": 1}),
        ]))
        self.assertEqual([e.line_number for e in errors], [3])
        self.assertEqual(records[0].response_headers["Retry-After"], "30")
        self.assertNotIn("SYNTHETIC", json.dumps(records[0].response_headers))
        self.assertEqual(records[1].response_headers, {})

    def test_short_id_survives_loading(self):
        records, _ = load_text(good(response_body={"id": "ord_8f3a12"}))
        self.assertEqual(records[0].response_body, {"id": "ord_8f3a12"})

    def test_sample_file(self):
        records, errors = load_logs(ROOT / "sample_data" / "logs.jsonl", CONFIG)
        self.assertEqual(errors, [])
        self.assertEqual(len(records), 6)
        self.assertNotIn("SYNTHETIC", json.dumps([asdict(r) for r in records]))
        self.assertIn("ord_8f3a12", records[0].url)


class ConfigAndModelTests(unittest.TestCase):
    def test_key_from_env_never_in_repr(self):
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "SYNTHETIC-KEY-xyz"}):
            cfg = load_config(env_path=Path("/nonexistent/.env"))
        self.assertTrue(cfg.llm_enabled)
        self.assertNotIn("SYNTHETIC", repr(cfg))

    def test_env_file_parsing(self):
        with tempfile.TemporaryDirectory() as d:
            env = Path(d) / ".env"
            env.write_text('# c\nOTHER=1\nGROQ_API_KEY="SYNTHETIC-FROM-FILE"\n')
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("GROQ_API_KEY", None)
                cfg = load_config(env_path=env)
        self.assertEqual(cfg.api_key, "SYNTHETIC-FROM-FILE")

    def test_no_key_means_llm_off(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GROQ_API_KEY", None)
            cfg = load_config(env_path=Path("/nonexistent/.env"))
        self.assertFalse(cfg.llm_enabled)

    def test_diagnosis_validates(self):
        ok = Diagnosis("auth", False, 0.9, "e", "f", "rule")
        self.assertEqual(ok.to_dict()["spec_reference"], {"operation": None, "field": None})
        with self.assertRaises(ValueError):
            Diagnosis("bogus", False, 0.9, "e", "f", "rule")
        with self.assertRaises(ValueError):
            Diagnosis("auth", False, 1.5, "e", "f", "rule")


if __name__ == "__main__":
    unittest.main()
