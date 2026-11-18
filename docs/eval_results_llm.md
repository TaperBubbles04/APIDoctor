# Evaluation results: mode `llm`

- Model: `openai/gpt-oss-120b`
- PROMPT_VERSION: `2`
- Cases: 20 (labeled, SYNTHETIC data)
- Real LLM calls this run: 23 (limit 25); cache hits: 0
- Skipped (quota): 0

> 20 labeled synthetic cases are far too few for statistical claims. Use these numbers to spot failure patterns, not as accuracy figures.

## Group: clear (12 cases, 12 scored, 0 skipped)

| metric | correct | out of scored |
|---|---|---|
| category | 12 | 12 |
| retryable | 12 | 12 |
| both | 12 | 12 |

## Group: ambiguous (5 cases, 5 scored, 0 skipped)

| metric | correct | out of scored |
|---|---|---|
| category | 5 | 5 |
| retryable | 5 | 5 |
| both | 5 | 5 |

## Group: trap (3 cases, 3 scored, 0 skipped)

| metric | correct | out of scored |
|---|---|---|
| category | 3 | 3 |
| retryable | 2 | 3 |
| both | 2 | 3 |

## Mismatches (category and/or retryable wrong)

- `t02` (trap): expected rate_limit/retryable=False, got rate_limit/retryable=True, source llm: The API responded with HTTP 429 and a message that the daily quota is exhausted, which matches the rate‑limit response defined in the spec (...
