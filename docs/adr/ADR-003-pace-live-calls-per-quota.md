# ADR-003: Pace live model calls per quota and model, off deadlines and latency

- Status: Accepted (PR #18)
- Date: 2026-11-19

## Context

Gemini's free tier allows 15 requests per minute per model and project for
Flash-Lite, and OpenRouter's `:free` models allow 20 per minute per key. One
ticket makes several model calls in a row (triage, coordinator turns,
subagents, reviewer), so a few tickets tripped the quota. The adapter retried
the 429s, but the retries ran out and the tickets turned into handoffs.
Waiting for a slot also ate into the agent's turn and request timeouts and the
eval runner's case timeout, so a ticket could time out while doing no work.

## Decision

- **Pace on our side.** `triagewright.llm.ratelimit.RateLimiter` is a sliding
  window over admissions: at most `per_minute` calls in any 61 s window. A
  slot is charged only when a call is admitted, so a caller cancelled while
  waiting costs nothing.
- **One limiter per model and quota.** `RateLimits.shared` is a process-wide
  registry keyed by `Settings.quota_id()` (provider, endpoint, and a hash of
  the key, never the key). The API, CLI, eval cases, and judge share it.
  Defaults come from `config.FREE_TIER_RPM`; `TW_LLM_REQUESTS_PER_MINUTE`
  overrides them, and `0` turns pacing off.
- **Waits stay off deadlines.** The limiter reports each wait to the current
  `triagewright.pacing.CallBudget` before sleeping. The budget moves its
  `asyncio.Timeout` back by the wait and tells the enclosing budget to do the
  same, so turn, request, and eval-case timeouts don't count queueing. A
  deadline that has already passed is never moved.
- **A cap on waiting.** `LoopConfig.max_pacing_s` (default 120 s) bounds the
  total wait per request. `RequestRun._pace` refuses a wait beyond it with
  `PacingRefusedError`, and the ticket is handed off.
- **Waits stay out of latency.** `RequestRun.call_model` reports each
  request's waits (`RequestTelemetry.record_pacing`, failed calls included).
  `RequestMetrics.latency_ms` excludes `paced_ms`, which is reported on its
  own. TTFT still includes the first call's wait.
- **Retries respect the deadline.** The adapter uses Gemini's `RetryInfo`
  delay when there is no `Retry-After` header, and skips a retry that can't
  finish before the deadline.

## Consequences

- Measured on 10 `happy-*` cases (Gemini Flash-Lite): no 429s, p50 3.6 s,
  p95 9.4 s, $0.00.
- A burst of tickets now queues instead of failing, and latency figures
  describe the agent's work, not the provider's quota.
- A request can take up to `max_pacing_s` longer in wall time than its
  latency shows. Anyone reading metrics must look at `paced_ms` too.
- Keys from one Gemini project share a quota but get separate windows here,
  because a key hash can't show which project it belongs to;
  `TW_LLM_REQUESTS_PER_MINUTE` can split the quota by hand.
- Pacing is per process. Two processes on one key don't share a window.

## Alternatives

- **Retry 429s only.** Rejected: retries ran out under a burst and turned into
  handoffs, and each failed attempt still used quota.
- **Space out eval cases only (`--min-interval`).** Kept as an eval option,
  but not enough: it doesn't cover the API or the calls inside one case.
- **Count waits in the timeouts and latency.** Rejected: queueing would trip
  timeouts with no work done and make latency gates measure the quota.
- **A cross-process limiter (file lock or a shared store).** Not needed for one
  local process and its eval runner; it would add a dependency.
