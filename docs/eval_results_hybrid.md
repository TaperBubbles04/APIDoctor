# Evaluation results: mode `hybrid`

- Model: `openai/gpt-oss-120b`
- PROMPT_VERSION: `2`
- Cases: 20 (labeled, SYNTHETIC data)
- Real LLM calls this run: 0 (limit 25); cache hits: 5
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
| category | 1 | 3 |
| retryable | 0 | 3 |
| both | 0 | 3 |

## Mismatches (category and/or retryable wrong)

- `t01` (trap): expected validation/retryable=False, got server_error/retryable=True, source rule: The server failed (500). This is usually temporary and not caused by the request.
- `t02` (trap): expected rate_limit/retryable=False, got rate_limit/retryable=True, source rule: Too many requests (429), but the server did not say how long to wait.
- `t03` (trap): expected rate_limit/retryable=True, got auth/retryable=False, source rule: The credentials were understood but do not have permission for this action.
