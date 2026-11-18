"""Evaluation: compare rules-only, LLM-only and rules+LLM on labeled cases."""
import argparse
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from apidoctor.config import PROJECT_ROOT, Config, load_config
from apidoctor.engine import ask_llm, diagnose
from apidoctor.llm.base import get_provider
from apidoctor.llm.limited import LimitedCachedProvider
from apidoctor.llm.prompt import PROMPT_VERSION
from apidoctor.loader import parse_record
from apidoctor.models import CATEGORIES, Diagnosis, LogRecord
from apidoctor.rules import apply_rules
from apidoctor.spec import excerpt_for, load_spec

MODES = ("rules", "llm", "hybrid")
GROUPS = ("clear", "ambiguous", "trap")
TOO_FEW = ("20 labeled synthetic cases are far too few for statistical claims. "
           "Use these numbers to spot failure patterns, not as accuracy figures.")


@dataclass
class Case:
    id: str
    group: str
    record: LogRecord
    expected_category: str
    expected_retryable: bool


def load_cases(path: Path, config: Config) -> list[Case]:
    cases, seen = [], set()
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            case_id, group, expected = obj["id"], obj["group"], obj["expected"]
            category, retryable = expected["category"], expected["retryable"]
        except (ValueError, KeyError, TypeError):
            raise ValueError(f"{Path(path).name} line {number}: not a valid case object") from None
        if case_id in seen:
            raise ValueError(f"duplicate case id {case_id!r}")
        if group not in GROUPS:
            raise ValueError(f"case {case_id}: bad group {group!r}")
        if category not in CATEGORIES:
            raise ValueError(f"case {case_id}: bad expected category {category!r}")
        if not isinstance(retryable, bool):
            raise ValueError(f"case {case_id}: expected retryable must be true or false")
        record, problem = parse_record(obj.get("record"), config)
        if problem:
            raise ValueError(f"case {case_id}: record problem: {problem}")
        seen.add(case_id)
        cases.append(Case(case_id, group, record, category, retryable))
    return cases


# ---- the provider wrapper: cache, call limit, pause --------------------------------------

class FakeProvider:
    """Plumbing test only: keyword guesses, no model. Results with it mean nothing."""

    def diagnose(self, record, spec_excerpt, rule_hint) -> Diagnosis:
        text = json.dumps(record.response_body).lower()
        if any(w in text for w in ("duplicate", "stale", "conflict")):
            category = "conflict"
        elif any(w in text for w in ("rate", "quota", "too many")):
            category = "rate_limit"
        elif any(w in text for w in ("invalid", "field")):
            category = "validation"
        else:
            category = "unknown"
        return Diagnosis(category, category == "rate_limit", 0.5,
                         "Fake provider keyword guess (not a real model).", "None: this is a fake.", "llm")


# ---- running the three modes ---------------------------------------------------------------

def _row(case: Case, category, retryable, source, explanation, skipped=False) -> dict:
    return {"id": case.id, "group": case.group,
            "expected_category": case.expected_category, "expected_retryable": case.expected_retryable,
            "got_category": category, "got_retryable": retryable,
            "source": source, "explanation": explanation, "skipped": skipped}


def _skipped(case: Case) -> dict:
    return _row(case, None, None, "none", "skipped (quota)", skipped=True)


def run_rules(cases, spec, config) -> list[dict]:
    rows = []
    for case in cases:
        result = apply_rules(case.record, spec, config.timeout_threshold_ms)
        if result.outcome == "diagnosed":
            d = result.diagnosis
            rows.append(_row(case, d.category, d.retryable, d.source, d.explanation))
        elif result.outcome == "needs_llm":
            rows.append(_row(case, "unknown", False, "rule", f"rules cannot decide: {result.reason}"))
        else:
            rows.append(_row(case, "not_a_failure", False, "rule", "rules say this was not a failure"))
    return rows


def run_llm(cases, spec, config, provider: LimitedCachedProvider, sleep) -> list[dict]:
    rows = []
    for case in cases:
        provider.refused = False
        answer = ask_llm(case.record, spec, provider, config, sleep)   # ignores the rules entirely
        if provider.refused:
            rows.append(_skipped(case))
        elif isinstance(answer, Diagnosis):
            rows.append(_row(case, answer.category, answer.retryable, "llm", answer.explanation))
        else:
            rows.append(_row(case, "unknown", False, "llm", f"LLM failed (reason: {answer})"))
    return rows


def run_hybrid(cases, spec, config, provider: LimitedCachedProvider, sleep) -> list[dict]:
    rows = []
    for case in cases:
        provider.refused = False
        d = diagnose(case.record, spec, provider, config, sleep)
        if provider.refused:
            rows.append(_skipped(case))
        elif d is None:
            rows.append(_row(case, "not_a_failure", False, "rule", "rules say this was not a failure"))
        else:
            rows.append(_row(case, d.category, d.retryable, d.source, d.explanation))
    return rows


# ---- scoring and reports -------------------------------------------------------------------

def score(rows: list[dict]) -> dict:
    """Per group: counts only. Skipped rows are neither right nor wrong."""
    out = {}
    for group in GROUPS:
        in_group = [r for r in rows if r["group"] == group]
        scored = [r for r in in_group if not r["skipped"]]
        cat = [r["got_category"] == r["expected_category"] for r in scored]
        ret = [r["got_retryable"] == r["expected_retryable"] for r in scored]
        out[group] = {
            "size": len(in_group), "scored": len(scored), "skipped": len(in_group) - len(scored),
            "category": sum(cat), "retryable": sum(ret),
            "both": sum(c and r for c, r in zip(cat, ret)),
        }
    return out


def _one_line(text: str, limit: int = 140) -> str:
    text = " ".join(str(text).split()).replace("|", "/")
    return text if len(text) <= limit else text[:limit] + "..."


def render_report(mode: str, rows: list[dict], info: dict) -> str:
    scores = score(rows)
    lines = [f"# Evaluation results: mode `{mode}`", ""]
    if info["fake"]:
        lines += ["**FAKE PROVIDER: these numbers test the plumbing only and mean nothing.**", ""]
    lines += [
        f"- Model: `{info['model']}`" + (" (not used in rules mode)" if mode == "rules" else ""),
        f"- PROMPT_VERSION: `{PROMPT_VERSION}`",
        f"- Cases: {len(rows)} (labeled, SYNTHETIC data)",
        f"- Real LLM calls this run: {info['real_calls']} (limit {info['max_calls']}); cache hits: {info['cache_hits']}",
        f"- Skipped (quota): {sum(r['skipped'] for r in rows)}",
        "", f"> {TOO_FEW}", "",
    ]
    for group in GROUPS:
        s = scores[group]
        lines += [
            f"## Group: {group} ({s['size']} cases, {s['scored']} scored, {s['skipped']} skipped)", "",
            "| metric | correct | out of scored |", "|---|---|---|",
            f"| category | {s['category']} | {s['scored']} |",
            f"| retryable | {s['retryable']} | {s['scored']} |",
            f"| both | {s['both']} | {s['scored']} |", "",
        ]
    lines += ["## Mismatches (category and/or retryable wrong)", ""]
    wrong = [r for r in rows if not r["skipped"] and not (
        r["got_category"] == r["expected_category"] and r["got_retryable"] == r["expected_retryable"])]
    for r in wrong:
        lines.append(
            f"- `{r['id']}` ({r['group']}): expected {r['expected_category']}/retryable={r['expected_retryable']}, "
            f"got {r['got_category']}/retryable={r['got_retryable']}, source {r['source']}: "
            f"{_one_line(r['explanation'])}")
    if not wrong:
        lines.append("- none")
    skipped = [r["id"] for r in rows if r["skipped"]]
    if skipped:
        lines += ["", "## Skipped (quota), not scored", "", ", ".join(f"`{i}`" for i in skipped)]
    return "\n".join(lines) + "\n"


def write_results(mode: str, rows: list[dict], info: dict, out_dir: Path, suffix: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / f"eval_results_{mode}{suffix}.md"
    md.write_text(render_report(mode, rows, info), encoding="utf-8")
    data = {"mode": mode, **info, "n_cases": len(rows), "prompt_version": PROMPT_VERSION,
            "scores": score(rows), "rows": rows}
    (out_dir / f"eval_results_{mode}{suffix}.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    return md


def _cell(s: dict, key: str) -> str:
    extra = f" (+{s['skipped']} skipped)" if s["skipped"] else ""
    return f"{s[key]}/{s['scored']}{extra}"


def render_summary(out_dir: Path, suffix: str) -> str:
    lines = ["| mode | clear both | ambiguous both | trap both | all: category | all: retryable | all: both |",
             "|---|---|---|---|---|---|---|"]
    notes = []
    for mode in MODES:
        path = out_dir / f"eval_results_{mode}{suffix}.json"
        if not path.is_file():
            lines.append(f"| {mode} | not run | | | | | |")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        sc = data["scores"]
        total = {k: sum(sc[g][k] for g in GROUPS) for k in ("scored", "skipped", "category", "retryable", "both")}
        lines.append(f"| {mode} | {_cell(sc['clear'], 'both')} | {_cell(sc['ambiguous'], 'both')} | "
                     f"{_cell(sc['trap'], 'both')} | {_cell(total, 'category')} | {_cell(total, 'retryable')} | "
                     f"{_cell(total, 'both')} |")
        notes.append(f"- {mode}: model `{data['model']}`, PROMPT_VERSION `{data['prompt_version']}`, "
                     f"{data['n_cases']} cases, {data['real_calls']} real LLM calls")
    fake = "**FAKE PROVIDER: plumbing test only.**\n\n" if suffix else ""
    return f"# Evaluation summary\n\n{fake}" + "\n".join(lines) + "\n\n" + "\n".join(notes) + f"\n\n> {TOO_FEW}\n"


# ---- command line ----------------------------------------------------------------------------

def main(argv=None, sleep: Callable[[float], None] = time.sleep) -> int:
    parser = argparse.ArgumentParser(description="Evaluate rules / LLM / hybrid on labeled cases.")
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--max-llm-calls", type=int, default=25)
    parser.add_argument("--fake", action="store_true", help="use a built-in fake provider (no key, no network)")
    parser.add_argument("--summarize", action="store_true", help="combine the three result files")
    parser.add_argument("--cases", default=str(PROJECT_ROOT / "sample_data" / "labeled_cases.jsonl"))
    parser.add_argument("--spec", default=str(PROJECT_ROOT / "sample_data" / "spec.yaml"))
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "docs"))
    parser.add_argument("--cache-path", default=None)
    args = parser.parse_args(argv)

    out_dir, suffix = Path(args.out_dir), "_fake" if args.fake else ""
    if args.summarize:
        text = render_summary(out_dir, suffix)
        (out_dir / f"eval_summary{suffix}.md").write_text(text, encoding="utf-8")
        print(text)
        return 0
    if not args.mode:
        parser.error("--mode is required (or use --summarize)")

    config = load_config()
    if args.fake:
        config = replace(config, llm_model="fake-model", delay_between_llm_calls_s=0.0)
    try:
        cases = load_cases(Path(args.cases), config)
        spec = load_spec(Path(args.spec))
        provider = None
        if args.mode != "rules":
            inner = FakeProvider() if args.fake else get_provider(config)
            if inner is None:
                print("No GROQ_API_KEY found (environment or .env): cannot run the LLM modes. Nothing was sent.")
                return 1
            default_cache = PROJECT_ROOT / ".cache" / f"llm_cache{suffix}.json"
            provider = LimitedCachedProvider(inner, config, Path(args.cache_path or default_cache),
                                             args.max_llm_calls, sleep)
    except ValueError as problem:
        print(f"Error: {problem}")
        return 1

    if args.mode == "rules":
        rows = run_rules(cases, spec, config)
    elif args.mode == "llm":
        rows = run_llm(cases, spec, config, provider, sleep)
    else:
        rows = run_hybrid(cases, spec, config, provider, sleep)
    info = {"model": config.llm_model, "fake": args.fake, "max_calls": args.max_llm_calls,
            "real_calls": provider.real_calls if provider else 0,
            "cache_hits": provider.cache_hits if provider else 0}
    path = write_results(args.mode, rows, info, out_dir, suffix)
    print(render_report(args.mode, rows, info))
    print(f"Written: {path}")
    return 0
