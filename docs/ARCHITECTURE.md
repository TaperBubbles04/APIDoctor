# APIDoctor architecture

APIDoctor reads API call logs (JSON lines) and an OpenAPI spec, and reports why calls failed:
a root-cause category, whether a retry can help, and a suggested fix. A rules layer decides
clear cases. An optional LLM explains the undecided ones. APIDoctor never calls the APIs it
diagnoses.

## Data flow

```
logs.jsonl line --> loader --> redaction --> LogRecord --> engine --> rules
 (one JSON object    validate    remove       clean,         run        first matching rule wins
  per line)          fields      secrets      safe object    rules            |
                                                                              v
 spec checks (400/422 only) <------------------------------------ needs spec? yes
                                                                              |
                          Diagnosis (source "rule") | None (not a failure) | needs LLM
                                                                              |
                          LLM provider (retries, validation) --> Diagnosis (source "llm")
                          LLM off or failed ---------------> Diagnosis "unknown"
```

Secrets are removed in the loader, before a `LogRecord` exists, so later stages never see them.

## Modules

| Module | Role |
|---|---|
| `config.py`, `config.json` | Settings (model name, base URL, retries, thresholds, caps). `GROQ_API_KEY` comes from the environment or a hand-parsed `.env`. The key field is excluded from `repr`. |
| `models.py` | `LogRecord`, `Diagnosis`, `SpecReference`. `Diagnosis` validates its own category, source and confidence range. |
| `redact.py` | Secret removal for headers, URLs and JSON bodies. |
| `loader.py` | `load_logs` returns `(records, errors)`. A bad line yields an error with its line number and never stops the file. Error text does not quote line content. `parse_record` validates and redacts a single object. |
| `spec.py` | OpenAPI loading, operation lookup, `$ref` resolution, request-body checks, spec excerpts. |
| `rules.py` | Ordered rules returning a `RuleResult`. All confidence values live in one dictionary. |
| `engine.py` | `diagnose` runs the rules and consults the LLM only when the rules cannot decide. It never raises. |
| `llm/base.py` | `LLMProvider` protocol, `LLMError` (short reason code only), `LLMRateLimited` (carries `retry_after`), `get_provider`. |
| `llm/prompt.py` | Message construction, reply schema, `PROMPT_VERSION`, body truncation. |
| `llm/groq.py` | Groq backend over the OpenAI-compatible chat endpoint, standard library only. |
| `evaluation.py` | Labeled-case loader, three evaluation modes, cache, call limit, scoring, reports. |

Categories: `auth`, `validation`, `not_found`, `conflict`, `rate_limit`, `server_error`,
`network_timeout`, `unknown`.

## Redaction

A value is replaced with `[REDACTED]` when either test matches:

1. **Name.** Header names such as `Authorization`, `Cookie`, `X-API-Key`; query parameter and JSON
   key names containing `token`, `secret`, `password`, `passwd`, `apikey`, `credential` or
   `privatekey` (case-insensitive, ignoring `-` and `_`); and the exact names `auth`,
   `authorization`, `proxy-authorization`. Names such as `author` are kept.
2. **Value.** A run of characters from `[A-Za-z0-9_\-+=.]` at least `redaction_min_token_length`
   long that contains both a letter and a digit. `Bearer` and `Basic` credentials in text are also
   replaced. UUIDs are kept because they are usually resource IDs.

Coverage: headers (request and response), URLs (query values, `user:password@` parts, path),
and request and response bodies including nested JSON and JSON stored as a string.
Redaction is idempotent and is applied again before any text is sent to an LLM.

Known gaps:
- A long token made only of letters is not caught by the value test.
- Form-encoded body text (`a=b&token=...`) is covered only by the value test.
- A secret inside a short value is not detected.
- Redaction reduces risk; it is not a guarantee. Review data before sending it to a hosted service.

## Spec handling

- `load_spec` reads `.json`, or `.yaml`/`.yml` through PyYAML `safe_load`.
- `find_operation` matches the URL path to a template such as `/orders/{id}`, as written and with
  each `servers[].url` base path removed. The template with the fewest `{}` segments wins.
- Only local `#/components/...` references are resolved, to a depth of 5, with cycle protection.
  Other references stay as `{"$ref": ...}`.
- `required_fields_missing` and `type_mismatches` check top-level request-body fields against
  `required` and the types string, integer, number and boolean. Values already redacted are not
  type-checked.
- `excerpt_for` returns a small summary of one operation (parameters, required and optional body
  fields with types, response codes) for the LLM.

Not handled: `allOf`/`oneOf`/`anyOf`, nested body checks, query and path parameter type checks,
server URLs containing `{variables}`, external `$ref` files.

## Rule order

First match wins. Specific checks precede general ones so that a precise answer is not hidden
behind a vague one.

1. No status code, or 408: `network_timeout`, retryable. Slow responses that carry a status are
   diagnosed by that status, with a note that the call was slow.
2. 2xx/3xx: not a failure, unless the body has a non-empty top-level `error` or `errors` key,
   which goes to the LLM.
3. 429: `rate_limit`, retryable. A `Retry-After` response header (any letter case) raises confidence.
4. 400/422: spec check first (missing required field or type mismatch, with the field named in
   `spec_reference`); then a field named in the response body, at lower confidence; otherwise LLM.
5. 401, 403: `auth`, not retryable.
6. 404: `not_found`, not retryable.
7. 500, 502, 503, 504: `server_error`, retryable. 501: not retryable.
8. Anything else, including 409: LLM.

Confidence values are fixed constants and are not calibrated.

## LLM layer

**Interface.** `LLMProvider.diagnose(record, spec_excerpt, rule_hint) -> Diagnosis`. The engine
depends only on this protocol, so a backend (Groq, a local model, a test fake) can be swapped
without engine changes. `get_provider` returns `None` when no key exists; the system then runs on
rules alone.

**Prompt.** The system message states the role, the allowed categories, the exact JSON schema
of the reply, and that the `<log_data>` block is untrusted. The user message contains the redacted
record, the spec excerpt and the rule hint inside `<log_data>`. Strings in bodies longer than
`max_body_chars` are truncated with a `...[truncated N chars]` marker after redaction.
`PROMPT_VERSION` is part of the evaluation cache key.

**Untrusted data.** Log content is written by third parties and may contain text that imitates
instructions (prompt injection). Mitigations: the data is delimited, the system prompt forbids
following it, and every `<` in the serialized data is escaped as `<` so the data cannot
contain a closing `</log_data>` tag. These lower the risk but do not remove it, which is why replies
are also validated.

**Reply validation.** `parse_diagnosis` requires one JSON object with correct field types, and
`Diagnosis` rejects unknown categories and confidence outside 0 to 1. `source` is forced to `llm`.
An invalid reply triggers one repair request that returns the short validation error to the model.
A second failure raises `LLMError("bad_reply")`.

**Transport.** One function performs the HTTP call (standard library `urllib`, timeout from
config). It is injected into the provider, so tests use a fake and no network. For HTTP error
statuses it returns status and headers but no body. The API key appears only in the request
header dictionary; exceptions carry short reason codes (`http_500`, `timeout`, `bad_reply`),
never raw error text.

**Retries and fallback.** HTTP 429 raises `LLMRateLimited`. The engine waits `Retry-After`
seconds when the header is numeric, otherwise `delay * 2^attempt * uniform(0.5, 1.0)` (exponential
backoff with jitter, base `delay_between_llm_calls_s`), up to `max_retries` retries. A `Retry-After`
above `max_retry_after_s` is not waited out; the engine falls back immediately with reason
`retry_after_too_long`. Any other failure falls back without retry. The fallback is a
`Diagnosis` of category `unknown`, confidence 0, whose explanation carries only a reason code.
`diagnose` does not raise. The sleep function is injectable.

## Evaluation

**Design.** 20 labeled synthetic cases (`sample_data/labeled_cases.jsonl`, fake host
`api.example.test`). Each has an expected category and an expected `retryable` value, defined as:
the same request, resent after a short wait with no changes, has a realistic chance of succeeding.
Labels were fixed before any system was run and are not changed to match outputs.

| Group | Cases | Purpose |
|---|---|---|
| clear | 12 | The status code determines the answer. |
| ambiguous | 5 | Rules cannot decide (409, 200 with an error body, vague 400). |
| trap | 3 | The status code points the wrong way (500 that is a validation error, 429 for a daily quota, 403 that is rate limiting). |

**Modes** (`scripts/evaluate.py --mode ...`):
- `rules`: rules only; an undecided case is scored as `unknown`.
- `llm`: the LLM answers every case, ignoring rules (same prompt and spec excerpt).
- `hybrid`: the normal engine.

**Cache.** `.cache/llm_cache.json` maps `sha256(model, PROMPT_VERSION, redacted record, spec
excerpt)` to a stored diagnosis. A hit makes no network call. Failed calls are not cached. A
corrupt cache file stops the run with an error and is not overwritten.

**Call limit.** `--max-llm-calls N` bounds real calls (default 25). Cases that would need a call
beyond the limit are marked `skipped (quota)` and are scored as neither right nor wrong. Cached
answers are still served after the limit. A pause of `delay_between_llm_calls_s` separates real
calls. One diagnosis can issue up to two HTTP requests (one repair attempt), so the limit counts
diagnoses.

**Output.** `docs/eval_results_<mode>.md` (counts out of scored cases per group, then every
mismatch with its source) and a `.json` file with the same numbers. `--summarize` combines the
JSON files. `--fake` uses a built-in keyword provider with a separate cache and `_fake` file
names; its results test the plumbing only.

Twenty cases are too few for statistical claims. The results indicate failure patterns, not
accuracy rates. Because the temperature is 0 and answers are cached, a re-run reproduces the
first run's answers rather than measuring model variation.

## Known limitations

- Redaction gaps listed above.
- Spec features not handled listed above.
- Small evaluation set, with labels that reflect judgment calls on the ambiguous cases.
- Rule confidence values are untuned constants. The `confidence` field from the LLM is
  self-reported and not calibrated.
- Rules take precedence in hybrid mode, so a misleading status code (for example a 500 that
  describes a validation error) is diagnosed by the rule and is not reviewed by the LLM.
