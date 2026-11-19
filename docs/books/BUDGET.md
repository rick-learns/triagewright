# Budget

Cap: **$50.00** for product model/API calls (dev runs, evals, recordings). Warn at $35.00. Stop and ask at $50.00.
Reviewer and worker model usage (sol-loop, council, delegates) is tracked separately and only when the harness reports it; it is never estimated.

## Product API spend

| date | task | provider | what | spend (USD) | running total |
| --- | --- | --- | --- | --- | --- |
| 2026-11-19 | W0 | openrouter | key metadata check (`GET /api/v1/key`, no model call) | 0.00 | 0.00 |
| 2026-11-19 | W0.4 | gemini (free plan) | 2× `models.list`; 7 one-turn tool-call probes (about 85 tokens each; 3 OK, 3× 503, 1 timeout) | 0.00 | 0.00 |

## Harness usage (reviewers/workers)

| date | task | model | usage reported by harness |
| --- | --- | --- | --- |
