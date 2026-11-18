# APIDoctor

APIDoctor reads API call logs (JSON lines) together with an OpenAPI spec and explains why calls
failed: a root-cause category, whether a plain retry can help, and a suggested fix. A rules layer
decides the clear cases without any model. An optional LLM, used only for cases the rules cannot
decide, explains the rest. Results are grouped by operation and category. APIDoctor only reads logs;
it never calls the APIs it diagnoses.

## Why it exists

A pile of failed API calls rarely says what to do next. A 500 might be a transient outage worth
retrying, or it might be a validation error wearing the wrong status code. A 429 might clear in 30
seconds, or it might be a daily quota that no retry will fix. Reading logs by hand, field by field
against the spec, is slow. APIDoctor decides the routine cases with plain rules, which are free, instant and easy to audit, and uses a model only for the genuinely ambiguous ones. The evaluation below measures what that trade-off costs.

## How it works

```
logs.jsonl
   |
   v
loader + redaction    (validate each line, remove secrets)
   |
   v
rules (+ spec checks) --decided--> diagnosis ---+
   |                                            |
   | undecided                                  v
   v                                       grouped report
LLM (optional; retries, validated reply)   (by operation and category)
   |  success --> diagnosis ----------------^
   |  off or failed --> "unknown" ----------^
```

Secrets are removed when a log line is loaded, before any later step sees it. Details are in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quick start

Requires Python 3.10+. Rules-only mode needs no key and no network.

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/python -m apidoctor diagnose sample_data/logs.jsonl --spec sample_data/spec.yaml
```

Add `--json` for machine-readable output.

On Debian/Ubuntu, if `python3 -m venv venv` fails with an `ensurepip` error, install the venv package first (`sudo apt install python3-venv`).

**ROS users:** ROS 2 sets `PYTHONPATH`, which makes a virtual environment see system packages. Prefix
every command with `env -u PYTHONPATH`, for example
`env -u PYTHONPATH ./venv/bin/python -m apidoctor diagnose ...`.

## Using the LLM

The LLM is off by default. To turn it on:

1. Get a free API key from Groq and put it in a `.env` file (copy `.env.example`):
   `GROQ_API_KEY=your-key`. `.env` is gitignored; never commit it.
2. Add `--llm`:

```bash
./venv/bin/python -m apidoctor diagnose sample_data/logs.jsonl --spec sample_data/spec.yaml --llm
```

Without a key, `--llm` stops with a clear message and a non-zero exit code. Free-tier limits change, so
nothing here hardcodes them: the model name, call cap, retry and pause settings are in `config.json`,
and `--max-llm-calls N` limits real calls per run. Answers are cached in `.cache/llm_cache.json`, so
repeating a run costs nothing. Only redacted data is sent to the provider, and log content is treated as
untrusted data in the prompt.

## The web page

```bash
./venv/bin/python -m apidoctor serve
```

Then open http://127.0.0.1:8000. The server listens on localhost only. Click **Try with sample data** to
load and run a built-in synthetic example, or choose or paste a log file and a spec, optionally tick "Use LLM" (enabled only when a key exists), and read the grouped summary followed
by a table of failures. Requests are limited to 2 MB and 1000 log lines.

## Evaluation

`sample_data/labeled_cases.jsonl` holds 20 labeled, synthetic cases: 12 clear (the status code
tells the whole story), 5 ambiguous (409 conflicts, a 200 with an error body, vague 400s) and 3 traps
(the status code points the wrong way). The labels were fixed before any system was run and not changed afterwards. "Retryable" means: the same request, resent after a short wait with no changes, has
a realistic chance of succeeding. Counts below are cases with the right answer out of cases scored
(model `openai/gpt-oss-120b`, prompt version 2); full numbers are in `docs/eval_results_*.json`.

| mode | clear both | ambiguous both | trap both | all: category | all: retryable | all: both |
|---|---|---|---|---|---|---|
| rules | 12/12 | 0/5 | 0/3 | 13/20 | 16/20 | 12/20 |
| llm | 12/12 | 5/5 | 2/3 | 20/20 | 19/20 | 19/20 |
| hybrid | 12/12 | 5/5 | 0/3 | 18/20 | 17/20 | 17/20 |

- `rules`: rules only. Cases the rules cannot decide count as `unknown`.
- `llm`: the model answers every case and ignores the rules.
- `hybrid`: the normal pipeline, rules first and the model only for undecided cases.

**Finding.** Hybrid fixes every ambiguous case, but on the traps it inherits the rules' mistakes: a rule
decides first, so a 500 that is really a validation error, a 403 that is really rate limiting, and a 429
for a daily quota are diagnosed by their status code and never reach the model. The LLM-only run got two of
the three traps fully right. This is a small set, so these counts show failure patterns, not accuracy
rates, and no statistical claim is made. Because answers are cached at temperature 0, a re-run repeats the
first run's answers instead of measuring model variation.

Per-mode reports: [rules](docs/eval_results_rules.md), [llm](docs/eval_results_llm.md),
[hybrid](docs/eval_results_hybrid.md); combined table: [summary](docs/eval_summary.md). To reproduce,
run each mode (the LLM modes need a key and make real calls):

```bash
./venv/bin/python scripts/evaluate.py --mode rules
./venv/bin/python scripts/evaluate.py --mode llm
./venv/bin/python scripts/evaluate.py --mode hybrid
./venv/bin/python scripts/evaluate.py --summarize
```

Add `--fake` to test the script without a key or network (it uses a keyword stub; its numbers mean nothing).

## Limitations

- **Redaction gaps:** a long token made only of letters is not caught by the value test; form-encoded body
  text is covered only by the length-based test; a secret inside a short value is not detected. Review data
  before sending it to any hosted service.
- **Spec features not handled:** `allOf`/`oneOf`/`anyOf`, nested request-body checks, query and path
  parameter type checks, server URLs containing `{variables}`, external `$ref` files.
- **Small evaluation set:** 20 synthetic cases, with labels that reflect judgment calls on the ambiguous cases.
- **Uncalibrated confidence:** the rule confidence values are fixed constants, and the model's confidence is
  self-reported and not calibrated.
- **Rules take precedence:** a misleading status code is diagnosed by the rule and is not reviewed.

## Future work

- Let the LLM review rule answers when the response text contradicts the status code (the trap cases).
- A local Ollama backend behind the same provider interface.
- A larger evaluation on real-world logs, with repeated runs to measure variation.

## Project structure

```
apidoctor/          package: loader, redaction, spec handling, rules, engine, service, CLI, web page
apidoctor/llm/      provider interface, prompt, Groq backend, cache and call-limit wrapper
scripts/            evaluate.py (three-mode evaluation), try_llm.py (one real call, run by hand)
sample_data/        demo logs and spec, 20 labeled evaluation cases (all synthetic)
tests/              unit tests (no network; the LLM is always faked)
docs/               ARCHITECTURE.md and evaluation results
config.json         settings (model name, retries, limits); no secrets
```

## Running tests

```bash
./venv/bin/python -m unittest discover -s tests -v
```

## License

MIT. See [LICENSE](LICENSE).
