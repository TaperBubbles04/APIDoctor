"""Command line: python -m apidoctor diagnose LOGS --spec SPEC [--llm] [--json] [--max-llm-calls N]"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable

import yaml

from apidoctor.config import PROJECT_ROOT, load_config
from apidoctor.llm.base import get_provider
from apidoctor.llm.limited import LimitedCachedProvider
from apidoctor.loader import load_logs
from apidoctor.service import DiagnoseResult, build_report, diagnose_logs
from apidoctor.spec import load_spec

CACHE_PATH = PROJECT_ROOT / ".cache" / "llm_cache.json"   # holds redacted-data answers only; gitignored


def _short(text: str, limit: int = 100) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "..."


def _yes_no(flag: bool | None) -> str:
    return "mixed" if flag is None else ("yes" if flag else "no")


def render_human(result: DiagnoseResult, load_errors: list, llm_stats: dict | None) -> str:
    lines = []
    for e in load_errors:
        lines.append(f"load error: line {e.line_number}: {e.message}")
    if load_errors:
        lines.append("")
    lines.append("Failures")
    for r in result.results:
        d = r.record
        status = d.status_code if d.status_code is not None else "no-response"
        lines.append(f"  {d.method} {d.url} | {status} | {r.diagnosis.category} | "
                     f"retryable: {_yes_no(r.diagnosis.retryable)} | {_short(r.diagnosis.explanation)}")
    if not result.results:
        lines.append("  none")
    lines += ["", "Summary (operation, category, count, retryable)"]
    for g in result.groups:
        lines.append(f"  {g.count:>3}x  {g.operation}  {g.category}  retryable: {_yes_no(g.retryable)}")
    if not result.groups:
        lines.append("  none")
    lines += ["", f"{result.total_records} records, {len(result.results)} failures, "
                  f"{result.non_failures} not failures (skipped), {len(load_errors)} load errors"]
    if llm_stats is not None:
        lines.append(f"LLM: {llm_stats['real_calls']} real calls, {llm_stats['cache_hits']} cache hits, "
                     f"{result.llm_skipped} records skipped (call limit)")
    return "\n".join(lines)


def run_diagnose(args, sleep: Callable[[float], None]) -> int:
    config = load_config()
    try:
        spec = load_spec(Path(args.spec))
        if not isinstance(spec, dict):
            raise ValueError("spec must be a JSON or YAML object")
    except (OSError, ValueError, yaml.YAMLError) as problem:
        print(f"Error: cannot read spec ({type(problem).__name__}).", file=sys.stderr)
        return 2
    try:
        records, load_errors = load_logs(Path(args.logs), config)
    except OSError:
        print("Error: cannot read the logs file.", file=sys.stderr)
        return 2

    provider = None
    if args.llm:
        inner = get_provider(config)
        if inner is None:
            print("Error: --llm needs GROQ_API_KEY (environment or .env). Nothing was sent.", file=sys.stderr)
            return 2
        limit = args.max_llm_calls if args.max_llm_calls is not None else config.max_llm_calls_per_run
        try:
            provider = LimitedCachedProvider(inner, config, CACHE_PATH, limit, sleep)
        except ValueError as problem:
            print(f"Error: {problem}", file=sys.stderr)
            return 2

    result = diagnose_logs(records, spec, provider, config, sleep)
    if args.json:
        print(json.dumps(build_report(result, load_errors, provider), indent=2))
    else:
        stats = {"real_calls": provider.real_calls, "cache_hits": provider.cache_hits} if provider else None
        print(render_human(result, load_errors, stats))
    return 1 if not records else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="apidoctor", description="Diagnose why API calls failed.")
    commands = parser.add_subparsers(dest="command", required=True)
    d = commands.add_parser("diagnose", help="diagnose a JSON-lines log file")
    d.add_argument("logs", help="path to the .jsonl log file")
    d.add_argument("--spec", required=True, help="path to the OpenAPI spec (.yaml, .yml or .json)")
    d.add_argument("--llm", action="store_true", help="use the LLM for cases the rules cannot decide (needs a key)")
    d.add_argument("--json", action="store_true", help="print machine-readable JSON")
    d.add_argument("--max-llm-calls", type=int, default=None, metavar="N",
                   help="most real LLM calls this run (default: max_llm_calls_per_run in config.json)")
    commands.add_parser("serve", help="run the local web page on 127.0.0.1:8000")
    return parser


def main(argv=None, sleep: Callable[[float], None] = time.sleep) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "diagnose":
        return run_diagnose(args, sleep)
    if args.command == "serve":
        from apidoctor.web import serve   # imported here so 'diagnose' works without the web packages
        print("APIDoctor web page: http://127.0.0.1:8000  (this machine only; Ctrl+C to stop)")
        serve()
        return 0
    return 2
