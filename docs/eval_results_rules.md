# Evaluation results: mode `rules`

- Model: `openai/gpt-oss-120b` (not used in rules mode)
- PROMPT_VERSION: `2`
- Cases: 20 (labeled, SYNTHETIC data)
- Real LLM calls this run: 0 (limit 25); cache hits: 0
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
| category | 0 | 5 |
| retryable | 4 | 5 |
| both | 0 | 5 |

## Group: trap (3 cases, 3 scored, 0 skipped)

| metric | correct | out of scored |
|---|---|---|
| category | 1 | 3 |
| retryable | 0 | 3 |
| both | 0 | 3 |

## Mismatches (category and/or retryable wrong)

- `a01` (ambiguous): expected conflict/retryable=False, got unknown/retryable=False, source rule: rules cannot decide: status 409 has no rule
- `a02` (ambiguous): expected conflict/retryable=False, got unknown/retryable=False, source rule: rules cannot decide: status 409 has no rule
- `a03` (ambiguous): expected rate_limit/retryable=False, got unknown/retryable=False, source rule: rules cannot decide: 200 response but the body contains an error
- `a04` (ambiguous): expected validation/retryable=False, got unknown/retryable=False, source rule: rules cannot decide: 400/422 with no spec problem found and no field named in the response
- `a05` (ambiguous): expected rate_limit/retryable=True, got unknown/retryable=False, source rule: rules cannot decide: 400/422 with no spec problem found and no field named in the response
- `t01` (trap): expected validation/retryable=False, got server_error/retryable=True, source rule: The server failed (500). This is usually temporary and not caused by the request.
- `t02` (trap): expected rate_limit/retryable=False, got rate_limit/retryable=True, source rule: Too many requests (429), but the server did not say how long to wait.
- `t03` (trap): expected rate_limit/retryable=True, got auth/retryable=False, source rule: The credentials were understood but do not have permission for this action.
