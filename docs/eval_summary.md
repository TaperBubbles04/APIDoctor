# Evaluation summary

| mode | clear both | ambiguous both | trap both | all: category | all: retryable | all: both |
|---|---|---|---|---|---|---|
| rules | 12/12 | 0/5 | 0/3 | 13/20 | 16/20 | 12/20 |
| llm | 12/12 | 5/5 | 2/3 | 20/20 | 19/20 | 19/20 |
| hybrid | 12/12 | 5/5 | 0/3 | 18/20 | 17/20 | 17/20 |

- rules: model `openai/gpt-oss-120b`, PROMPT_VERSION `2`, 20 cases, 0 real LLM calls
- llm: model `openai/gpt-oss-120b`, PROMPT_VERSION `2`, 20 cases, 23 real LLM calls
- hybrid: model `openai/gpt-oss-120b`, PROMPT_VERSION `2`, 20 cases, 0 real LLM calls

> 20 labeled synthetic cases are far too few for statistical claims. Use these numbers to spot failure patterns, not as accuracy figures.
