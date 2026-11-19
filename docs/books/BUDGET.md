# Budget

Cap: **$50.00** for product model/API calls (dev runs, evals, recordings). Warn at $35.00. Stop and ask at $50.00.
Reviewer and worker model usage (sol-loop, council, delegates) is tracked separately and only when the harness reports it; it is never estimated.

## Product API spend

| date | task | provider | what | spend (USD) | running total |
| --- | --- | --- | --- | --- | --- |
| 2026-11-19 | W0 | openrouter | key metadata check (`GET /api/v1/key`, no model call) | 0.00 | 0.00 |
| 2026-11-19 | W0.4 | gemini (free plan) | 2× `models.list`; 7 one-turn tool-call probes (about 85 tokens each; 3 OK, 3× 503, 1 timeout) | 0.00 | 0.00 |
| 2026-11-19 | Q1–Q4 | gemini (free plan) | live smoke runs (happy-* ×10, several times), one full 54-case eval, single-case repros; free plan, no billing; daily per-model quota (500 requests) reached once | 0.00 | 0.00 |
| 2026-11-19 | Q4 | openrouter (`:free`) | 3-case spot check; all 429 (free-model rate limit) | 0.00 | 0.00 |
| 2026-11-19 | Q5/X2 | gemini (free plan) | one 54-case attempt on `gemini-3.5-flash-lite` (invalid: every model call got HTTP 429, daily quota used up); live runs on `gemini-3.1-flash-lite` (one partial run, 26 cases; one run stopped after 14 cases when 13 of 19 calls timed out); model probes | 0.00 | 0.00 |
| 2026-11-19 | X2 | openrouter (`:free`) | key metadata check (usage $0); tool-call probes; `gemma-4-31b-it:free` run stopped (429 upstream); 3-case probe and a 54-case run (in progress at this commit) on `nemotron-3-ultra-550b-a55b:free` | 0.00 | 0.00 |

## Harness usage (reviewers/workers)

| date | task | model | usage reported by harness |
| --- | --- | --- | --- |
