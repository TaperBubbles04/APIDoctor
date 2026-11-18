import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from apidoctor import evaluation as ev
from apidoctor.config import load_config
from apidoctor.llm.base import LLMError
from apidoctor.models import CATEGORIES, Diagnosis, LogRecord
from apidoctor.spec import find_operation, load_spec, required_fields_missing, type_mismatches

ROOT = Path(__file__).resolve().parent.parent
SPEC = load_spec(ROOT / "sample_data" / "spec.yaml")
BASE = load_config(env_path=Path("/nonexistent/.env"))
NO_KEY = replace(BASE, api_key=None, delay_between_llm_calls_s=3)
CASES_FILE = ROOT / "sample_data" / "labeled_cases.jsonl"

# The labels are fixed by the project owner. This pins them so an accidental edit fails loudly.
PINNED = {
    "c01": ("auth", False), "c02": ("auth", False), "c03": ("not_found", False),
    "c04": ("validation", False), "c05": ("validation", False), "c06": ("rate_limit", True),
    "c07": ("server_error", True), "c08": ("server_error", True), "c09": ("network_timeout", True),
    "c10": ("server_error", True), "c11": ("network_timeout", True), "c12": ("server_error", True),
    "a01": ("conflict", False), "a02": ("conflict", False), "a03": ("rate_limit", False),
    "a04": ("validation", False), "a05": ("rate_limit", True),
    "t01": ("validation", False), "t02": ("rate_limit", False), "t03": ("rate_limit", True),
}


class Scripted:
    """Fake provider: counts calls, answers with a fixed category."""
    def __init__(self, category="conflict"):
        self.calls, self.category = 0, category

    def diagnose(self, record, spec_excerpt, rule_hint):
        self.calls += 1
        return Diagnosis(self.category, False, 0.8, "scripted answer", "scripted fix", "llm")


class Failing:
    def diagnose(self, record, spec_excerpt, rule_hint):
        raise LLMError("http_500")


def case(cid, status=409, method="PUT", group="ambiguous", url=None, expected=("conflict", False), headers=None):
    record = LogRecord("t", method, url or f"https://api.example.test/v1/orders/{cid}", status, {},
                       None, {"error": "x"}, 10, headers or {})
    return ev.Case(cid, group, record, *expected)


def wrapper(inner, tmp, max_calls=25, sleeps=None, config=NO_KEY, name="cache.json"):
    return ev.LimitedCachedProvider(inner, config, Path(tmp) / name, max_calls,
                                    sleeps.append if sleeps is not None else (lambda s: None))


def row(group, ec, er, gc, gr, skipped=False):
    return {"id": "x", "group": group, "expected_category": ec, "expected_retryable": er,
            "got_category": gc, "got_retryable": gr, "source": "rule", "explanation": "e", "skipped": skipped}


class LabeledFile(unittest.TestCase):
    def test_loads_20_valid_cases(self):
        cases = ev.load_cases(CASES_FILE, BASE)
        self.assertEqual(len(cases), 20)
        self.assertEqual(len({c.id for c in cases}), 20)
        self.assertTrue(all(c.expected_category in CATEGORIES for c in cases))
        counts = {g: sum(c.group == g for c in cases) for g in ev.GROUPS}
        self.assertEqual(counts, {"clear": 12, "ambiguous": 5, "trap": 3})
        self.assertTrue(all(c.record.url.startswith("https://api.example.test/") for c in cases))

    def test_labels_are_the_fixed_ones(self):
        got = {c.id: (c.expected_category, c.expected_retryable) for c in ev.load_cases(CASES_FILE, BASE)}
        self.assertEqual(got, PINNED)

    def test_cases_that_must_violate_the_spec_do(self):
        by_id = {c.id: c.record for c in ev.load_cases(CASES_FILE, BASE)}
        for cid in ("c04", "c05", "t01"):
            record = by_id[cid]
            operation = find_operation(SPEC, record.method, record.url)
            self.assertIsNotNone(operation, cid)
            self.assertTrue(required_fields_missing(record, operation) or type_mismatches(record, operation), cid)

    def test_no_secret_survives_loading(self):
        text = json.dumps([c.record.request_headers for c in ev.load_cases(CASES_FILE, BASE)])
        self.assertNotIn("SYNTHETIC", text)

    def test_bad_case_files_are_rejected(self):
        good = json.loads(CASES_FILE.read_text().splitlines()[0])
        variants = {
            "bad json": "{nope",
            "bad group": {**good, "group": "easy"},
            "bad category": {**good, "expected": {"category": "banana", "retryable": False}},
            "retryable not bool": {**good, "expected": {"category": "auth", "retryable": "no"}},
            "bad record": {**good, "record": {"method": "GET"}},
        }
        with tempfile.TemporaryDirectory() as d:
            for name, variant in variants.items():
                p = Path(d) / "c.jsonl"
                p.write_text(variant if isinstance(variant, str) else json.dumps(variant))
                with self.assertRaises(ValueError, msg=name):
                    ev.load_cases(p, BASE)
            p.write_text(json.dumps(good) + "\n" + json.dumps(good))
            with self.assertRaises(ValueError):   # duplicate id
                ev.load_cases(p, BASE)


class RulesMode(unittest.TestCase):
    def setUp(self):
        self.cases = ev.load_cases(CASES_FILE, BASE)
        self.rows = ev.run_rules(self.cases, SPEC, BASE)
        self.scores = ev.score(self.rows)

    def test_clear_cases_are_all_right(self):
        self.assertEqual(self.scores["clear"], {"size": 12, "scored": 12, "skipped": 0,
                                                "category": 12, "retryable": 12, "both": 12})

    def test_needs_llm_cases_are_scored_as_unknown(self):
        ambiguous = [r for r in self.rows if r["group"] == "ambiguous"]
        self.assertEqual({r["got_category"] for r in ambiguous}, {"unknown"})
        self.assertEqual(self.scores["ambiguous"]["category"], 0)
        self.assertEqual(self.scores["ambiguous"]["both"], 0)

    def test_traps_fool_the_rules_as_designed(self):
        got = {r["id"]: (r["got_category"], r["got_retryable"]) for r in self.rows if r["group"] == "trap"}
        self.assertEqual(got, {"t01": ("server_error", True), "t02": ("rate_limit", True), "t03": ("auth", False)})
        self.assertEqual((self.scores["trap"]["category"], self.scores["trap"]["both"]), (1, 0))


class CacheTests(unittest.TestCase):
    def test_cache_hit_skips_provider(self):
        with tempfile.TemporaryDirectory() as d:
            inner, c = Scripted(), case("k1")
            w = wrapper(inner, d)
            first = w.diagnose(c.record, None, None)
            second = w.diagnose(c.record, None, None)
            self.assertEqual((inner.calls, w.real_calls, w.cache_hits), (1, 1, 1))
            self.assertEqual(first.to_dict(), second.to_dict())

    def test_cache_survives_between_runs(self):
        with tempfile.TemporaryDirectory() as d:
            c = case("k2")
            wrapper(Scripted(), d).diagnose(c.record, None, None)
            inner = Scripted()
            w2 = wrapper(inner, d)
            self.assertEqual(w2.diagnose(c.record, None, None).category, "conflict")
            self.assertEqual((inner.calls, w2.real_calls, w2.cache_hits), (0, 0, 1))

    def test_key_changes_with_model_prompt_version_and_excerpt(self):
        with tempfile.TemporaryDirectory() as d:
            c = case("k3")
            base = wrapper(Scripted(), d).key(c.record, {"a": 1})
            self.assertNotEqual(base, wrapper(Scripted(), d, config=replace(NO_KEY, llm_model="other")).key(c.record, {"a": 1}))
            self.assertNotEqual(base, wrapper(Scripted(), d).key(c.record, {"a": 2}))
            with mock.patch("apidoctor.llm.limited.PROMPT_VERSION", "99"):
                self.assertNotEqual(base, wrapper(Scripted(), d).key(c.record, {"a": 1}))

    def test_key_uses_the_redacted_record(self):
        with tempfile.TemporaryDirectory() as d:
            w = wrapper(Scripted(), d)
            a = case("k4", url="https://api.example.test/v1/orders/o?api_key=SYNTHETIC-AAAAAAAAAAAAAAAAAAAA")
            b = case("k4", url="https://api.example.test/v1/orders/o?api_key=SYNTHETIC-BBBBBBBBBBBBBBBBBBBB")
            self.assertEqual(w.key(a.record, None), w.key(b.record, None))

    def test_failed_calls_are_not_cached(self):
        with tempfile.TemporaryDirectory() as d:
            w = wrapper(Failing(), d)
            with self.assertRaises(LLMError):
                w.diagnose(case("k5").record, None, None)
            self.assertEqual(w.cache, {})
            self.assertFalse((Path(d) / "cache.json").exists())

    def test_corrupt_cache_file_is_reported_not_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "cache.json").write_text("{broken")
            with self.assertRaises(ValueError):
                wrapper(Scripted(), d)
            self.assertEqual((Path(d) / "cache.json").read_text(), "{broken")


class CallLimitAndPause(unittest.TestCase):
    def setUp(self):
        self.cases = [case(f"ord_{i}") for i in (1, 2, 3)]

    def test_max_calls_stops_early_and_marks_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            inner = Scripted()
            w = wrapper(inner, d, max_calls=2)
            rows = ev.run_llm(self.cases, SPEC, NO_KEY, w, lambda s: None)
            self.assertEqual([r["skipped"] for r in rows], [False, False, True])
            self.assertEqual((inner.calls, w.real_calls), (2, 2))
            self.assertIsNone(rows[2]["got_category"])
            s = ev.score(rows)["ambiguous"]
            self.assertEqual((s["size"], s["scored"], s["skipped"], s["both"]), (3, 2, 1, 2))   # skipped is not wrong

    def test_cache_is_still_served_after_the_limit(self):
        with tempfile.TemporaryDirectory() as d:
            ev.run_llm(self.cases, SPEC, NO_KEY, wrapper(Scripted(), d), lambda s: None)
            inner = Scripted()
            w = wrapper(inner, d, max_calls=0)
            rows = ev.run_llm(self.cases, SPEC, NO_KEY, w, lambda s: None)
            self.assertFalse(any(r["skipped"] for r in rows))
            self.assertEqual((inner.calls, w.cache_hits), (0, 3))

    def test_pause_only_between_real_calls(self):
        with tempfile.TemporaryDirectory() as d:
            sleeps = []
            ev.run_llm(self.cases, SPEC, NO_KEY, wrapper(Scripted(), d, sleeps=sleeps), lambda s: None)
            self.assertEqual(sleeps, [3, 3])   # 3 real calls, 2 pauses
            sleeps.clear()
            ev.run_llm(self.cases, SPEC, NO_KEY, wrapper(Scripted(), d, sleeps=sleeps), lambda s: None)
            self.assertEqual(sleeps, [])   # all cache hits: no pause

    def test_llm_failure_is_scored_wrong_not_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            rows = ev.run_llm(self.cases[:1], SPEC, NO_KEY, wrapper(Failing(), d), lambda s: None)
            self.assertFalse(rows[0]["skipped"])
            self.assertEqual(rows[0]["got_category"], "unknown")
            self.assertIn("http_500", rows[0]["explanation"])


class ModesDiffer(unittest.TestCase):
    def test_llm_mode_ignores_rules_hybrid_does_not(self):
        c = case("ord_9", status=404, method="GET", group="clear", expected=("not_found", False))
        with tempfile.TemporaryDirectory() as d:
            llm_inner, hybrid_inner = Scripted(), Scripted()
            ev.run_llm([c], SPEC, NO_KEY, wrapper(llm_inner, d, name="a.json"), lambda s: None)
            ev.run_hybrid([c], SPEC, NO_KEY, wrapper(hybrid_inner, d, name="b.json"), lambda s: None)
            self.assertEqual((llm_inner.calls, hybrid_inner.calls), (1, 0))

    def test_hybrid_rule_cases_do_not_use_quota_or_get_skipped(self):
        cases = [case("ord_1"), case("ord_8", status=404, method="GET", expected=("not_found", False)), case("ord_2")]
        with tempfile.TemporaryDirectory() as d:
            w = wrapper(Scripted(), d, max_calls=1)
            rows = ev.run_hybrid(cases, SPEC, NO_KEY, w, lambda s: None)
            self.assertEqual([r["skipped"] for r in rows], [False, False, True])
            self.assertEqual(rows[1]["source"], "rule")
            self.assertEqual(w.real_calls, 1)

    def test_hybrid_uses_llm_answer_for_ambiguous_case(self):
        with tempfile.TemporaryDirectory() as d:
            rows = ev.run_hybrid([case("ord_1")], SPEC, NO_KEY, wrapper(Scripted(), d), lambda s: None)
            self.assertEqual((rows[0]["got_category"], rows[0]["source"]), ("conflict", "llm"))


class ScoringAndReport(unittest.TestCase):
    ROWS = [
        row("clear", "auth", False, "auth", False),                 # both right
        row("clear", "auth", False, "auth", True),                  # category right, retryable wrong
        row("clear", "auth", False, "unknown", False),              # category wrong, retryable right
        row("clear", "auth", False, None, None, skipped=True),      # skipped
        row("trap", "rate_limit", True, "auth", False),             # both wrong
    ]

    def test_counts(self):
        s = ev.score(self.ROWS)
        self.assertEqual(s["clear"], {"size": 4, "scored": 3, "skipped": 1, "category": 2, "retryable": 2, "both": 1})
        self.assertEqual(s["trap"], {"size": 1, "scored": 1, "skipped": 0, "category": 0, "retryable": 0, "both": 0})
        self.assertEqual(s["ambiguous"]["size"], 0)

    def test_report_text(self):
        info = {"model": "some-model", "fake": False, "real_calls": 4, "cache_hits": 1, "max_calls": 25}
        text = ev.render_report("hybrid", self.ROWS, info)
        for needle in ("some-model", f"PROMPT_VERSION: `{ev.PROMPT_VERSION}`", "far too few for statistical claims",
                       "| both | 1 | 3 |", "| category | 2 | 3 |", "## Group: clear", "## Group: ambiguous",
                       "## Group: trap", "Mismatches", "Skipped (quota)", "SYNTHETIC"):
            self.assertIn(needle, text)
        self.assertNotIn("FAKE", text)
        self.assertIn("FAKE PROVIDER", ev.render_report("hybrid", self.ROWS, {**info, "fake": True}))


def run_main(*argv):
    out = io.StringIO()
    with mock.patch("apidoctor.evaluation.load_config", return_value=NO_KEY), contextlib.redirect_stdout(out):
        code = ev.main(list(argv), sleep=lambda s: None)
    return code, out.getvalue()


class CommandLine(unittest.TestCase):
    def test_fake_hybrid_writes_only_fake_files(self):
        with tempfile.TemporaryDirectory() as d:
            code, out = run_main("--mode", "hybrid", "--fake", "--out-dir", d, "--cache-path", f"{d}/c.json")
            self.assertEqual(code, 0)
            names = sorted(p.name for p in Path(d).iterdir())
            self.assertEqual(names, ["c.json", "eval_results_hybrid_fake.json", "eval_results_hybrid_fake.md"])
            self.assertIn("FAKE PROVIDER", out)

    def test_fake_default_cache_is_separate_from_the_real_cache(self):
        with tempfile.TemporaryDirectory() as d, mock.patch("apidoctor.evaluation.PROJECT_ROOT", Path(d)):
            code, _ = run_main("--mode", "llm", "--fake", "--out-dir", d, "--cases", str(CASES_FILE),
                               "--spec", str(ROOT / "sample_data" / "spec.yaml"))
            self.assertEqual(code, 0)
            self.assertTrue((Path(d) / ".cache" / "llm_cache_fake.json").is_file())
            self.assertFalse((Path(d) / ".cache" / "llm_cache.json").exists())

    def test_max_llm_calls_flag(self):
        with tempfile.TemporaryDirectory() as d:
            code, _ = run_main("--mode", "llm", "--fake", "--max-llm-calls", "3", "--out-dir", d,
                               "--cache-path", f"{d}/c.json")
            data = json.loads((Path(d) / "eval_results_llm_fake.json").read_text())
            self.assertEqual((code, data["real_calls"]), (0, 3))
            self.assertEqual(sum(r["skipped"] for r in data["rows"]), 17)

    def test_second_run_is_free_thanks_to_the_cache(self):
        with tempfile.TemporaryDirectory() as d:
            args = ("--mode", "llm", "--fake", "--out-dir", d, "--cache-path", f"{d}/c.json")
            run_main(*args)
            run_main(*args)
            data = json.loads((Path(d) / "eval_results_llm_fake.json").read_text())
            self.assertEqual((data["real_calls"], data["cache_hits"]), (0, 20))

    def test_llm_modes_need_a_key_but_rules_mode_does_not(self):
        with tempfile.TemporaryDirectory() as d:
            code, out = run_main("--mode", "llm", "--out-dir", d, "--cache-path", f"{d}/c.json")
            self.assertEqual(code, 1)
            self.assertIn("No GROQ_API_KEY", out)
            self.assertEqual(list(Path(d).iterdir()), [])
            code, _ = run_main("--mode", "rules", "--out-dir", d)
            self.assertEqual(code, 0)
            self.assertTrue((Path(d) / "eval_results_rules.md").is_file())
            self.assertFalse((Path(d) / "c.json").exists())

    def test_bad_cache_or_cases_give_a_clean_error(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "c.json").write_text("{broken")
            code, out = run_main("--mode", "llm", "--fake", "--out-dir", d, "--cache-path", f"{d}/c.json")
            self.assertEqual(code, 1)
            self.assertIn("Error:", out)

    def test_summarize_combines_the_modes(self):
        with tempfile.TemporaryDirectory() as d:
            run_main("--mode", "rules", "--fake", "--out-dir", d)
            run_main("--mode", "hybrid", "--fake", "--out-dir", d, "--cache-path", f"{d}/c.json")
            code, out = run_main("--summarize", "--fake", "--out-dir", d)
            self.assertEqual(code, 0)
            self.assertIn("| rules | 12/12 |", out)
            self.assertIn("| llm | not run |", out)
            self.assertIn("hybrid", out)
            self.assertTrue((Path(d) / "eval_summary_fake.md").is_file())


if __name__ == "__main__":
    unittest.main()
