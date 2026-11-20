# ADR-008: The eval report runs on OpenRouter `:free` Nemotron, judge included

- Status: Accepted (owner instruction, 2026-11-19: on free models only, use
  another free model when one runs out, and label every result)
- Date: 2026-11-19

## Context

ADR-002 made the Gemini API free tier the preferred live provider. On the day
of the report runs, it couldn't carry a full 54-case run (BUDGET, 2026-11-19):

- `gemini-3.5-flash-lite` used up its daily free quota, and a 54-case attempt
  got HTTP 429 on every model call. It's kept as an invalid run.
- `gemini-3.1-flash-lite` timed out too often: one run stopped after 14 cases,
  when 13 of 19 calls had timed out.
- On OpenRouter `:free`, `google/gemma-4-31b-it:free` (the OpenRouter default
  for agent roles) returned 429 from upstream.

`nvidia/nemotron-3-ultra-550b-a55b:free` passed a 3-case probe and then ran
all 54 cases. It is slower than Flash-Lite: in development run full5, one
model call had a p50 of 4.7 s and a p95 of 17.5 s, against a whole-ticket p95
of 9.4 s for Flash-Lite on 10 happy-path cases (ADR-003). A ticket makes about
five or six model calls in sequence.

## Decision

- The report's live runs use OpenRouter with
  `nvidia/nemotron-3-ultra-550b-a55b:free` for every role (`TW_MODEL_*`),
  the judge included. No code default changes: this is configuration, as
  ADR-002 intends for the fallback.
- Every result names its provider, model, commit, and budgets. Runs that
  failed for provider reasons are kept, marked invalid, and not reported as
  results.
- The report has two labeled runs on the same commit: the default budget
  (`total_timeout_s` 30 s, `turn_timeout_s` 20 s), and a long budget
  (120 s and 45 s) so the provider's latency doesn't decide the pass rate.
  Neither is presented as a trade-off of the other.
- Replay recordings come from the long-budget run.

## Consequences

- The judge is the same model as the agent. Self-preference is possible, so
  the report says so and a sample of the judge's verdicts is checked by hand.
- p95 latency reflects this provider's free queue. The report states the
  target as not met if it misses, with the per-call breakdown, and doesn't
  redefine the metric.
- A Gemini Flash-Lite rerun is possible once its quota resets. If it changes
  the results materially, the report is updated with both, labeled.
- Recording fingerprints include the model id, so replay needs the same
  `TW_MODEL_*` settings as the recording run (the replay defaults are the
  Gemini table).
- The free OpenRouter quota (about 1,000 requests a day on this key, R2) fits
  two full runs with little room to spare (R6).

## Alternatives

- **Wait for the Gemini quota.** Rejected for the headline: the owner asked
  not to idle on a quota. A rerun stays possible.
- **A different model per role on OpenRouter.** Not adopted: the other free
  models probed returned 429 or timed out, and one model for every role keeps
  the labels simple.
- **A different judge model.** Not adopted: no other free model was reliable
  that day. The hand check is the mitigation.
- **A paid model.** Rejected: $0 only (ADR-002).
