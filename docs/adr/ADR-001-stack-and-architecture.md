# ADR-001: Stack and architecture

- Status: Accepted (council round 5, 2026-11-19; PR #1)
- Date: 2026-11-19

## Context

triagewright is a local-first customer-support agent: triage, tool use over MCP,
grounded answers with citations, and escalation to a human. It must run on one
machine with `docker compose up`, stay under $50 in model spend, and
be testable in CI with no API keys. Ten lanes are built in parallel, so the
contracts between them must be fixed before fan-out.

## Decisions

### 1. Python 3.12, uv, Pydantic v2 contracts as the single source of truth

All cross-lane types live in `src/triagewright/contracts/`. They're frozen,
reject unknown fields, and use integer cents. JSON Schemas in `schemas/` are
generated from them (`scripts/export_schemas.py`), and a test fails if they're
stale. Lanes talk only through the Protocols in `contracts/ports.py`.

### 2. Model layer shaped like the Anthropic Messages API, served by OpenRouter

- `ModelRequest`/`ModelResponse` mirror Messages: content blocks (`text`,
  `tool_use`, `tool_result`), `stop_reason`, and usage including cache tokens.
  `Usage.input_tokens` excludes cache tokens; adapters normalize to that.
- **The owner chose OpenRouter as the only live provider** (2026-11-19).
  Adapters: `openrouter` and `replay`. The OpenRouter adapter translates
  to and from its OpenAI-compatible chat API and refuses model ids without
  `:free`; settings reject any other model id, with no opt-out. Defaults:
  `google/gemma-4-31b-it:free` for every role and
  `nvidia/nemotron-3-ultra-550b-a55b:free` for the judge, both with tool
  calling. Free models are rate-limited (20 requests/min, and a daily cap),
  so live runs are throttled and resumable.
- The default provider is `replay`, so nothing calls the network unless
  `.env` opts in. CI always replays. A missing recording is a hard error,
  never a live fallback.
- **Replay is deterministic by construction.** Recordings are stored per case
  (`evals/fixtures/recordings/<case_id>.jsonl`) and matched by
  `ModelRequest.fingerprint()`, which includes the model id. Identical
  fingerprints are served in recorded order. Everything a model sees is
  serialized with `common.canonical_json`. Every model-visible id (approval,
  escalation, refund, request) comes from `common.stable_id` over inputs
  such as the ticket id, never uuid4 or the wall clock. The clock is the
  case's `as_of`. Recording happens only after the MCP server (A), RAG (C),
  and guardrails (D) have landed, because their output is part of every
  fingerprint.

### 3. Coordinator plus isolated subagents (hub and spoke)

- The **coordinator** runs an explicit agentic loop. Each turn sends the
  transcript and checks `stop_reason`:
  - `tool_use`: run every requested tool and append the `tool_result` blocks.
  - `end_turn`: finish.
  - `max_tokens`: one continuation, then a safe fallback.
  - `pause_turn`: continue.
  - `refusal`: stop with `LoopStop.REFUSAL`.
- The loop has bounds: `max_turns`, a per-turn timeout, and a total timeout.
  Hitting any bound gives a safe fallback, which escalates to a human with a
  handoff summary. It never gives a half-finished action.
- Its tools are the four subagents plus the read-only business tools. The
  final answer is a forced `finalize` tool call (structured output through a
  tool schema), validated with Pydantic and retried once with the validation
  error. The model supplies only the reply, citation candidates, and a
  proposed status (`resolved`, `needs_info`, or `refused`). **Code builds
  `FinalResponse`:** `actions` and `escalation` come from executor and
  gateway records, and a pending approval or a created escalation overrides
  the proposed status.
- After the escalation reviewer decides to escalate, coordinator code (not a
  model) calls `create_escalation` from the `EscalationDecision`, with the
  ticket id from the trusted context. `create_escalation` is code-owned: the
  gateway accepts it only from `caller="coordinator"` for `ctx.ticket_id`,
  and the executor refuses it as a model proposal, so a model can't claim
  the ticket's one escalation first. It needs no customer scope, so
  anonymous tickets can still be handed off. It's idempotent per ticket, so
  the security handoff is decided before any other escalation.
  `LoopConfig.handoff_reserve_s` is held back from the total budget for it.
  If the write still fails, the reply says the request was logged for
  review and does not promise a follow-up time.
- Retries of retryable tool errors happen in one place: lane B's
  deterministic tool-call wrapper (`LoopConfig.tool_retries`, backoff from
  `retry_after_ms` or `retry_backoff_s`, zero in evals). Not the model, and
  not the gateway, so `ToolFault.times` counts raw gateway calls.
- **Subagents** (`triage`, `answer`, `action`, `escalation`) get only their
  typed input (`TriageInput`, `AnswerInput`, …), never the coordinator
  transcript. Each returns its typed output, validated before the coordinator
  sees it. Each subagent may run its own bounded tool loop. The answer
  subagent may use `search_kb` only. The action subagent may call the
  read-only tools in `ActionInput.allowed_tools` directly. It reaches writes
  only through `ActionExecutor.propose`.

### 4. Guardrails are code, not prompts

- **Customer scope.** Every tool call carries a `CallContext` set by code
  from the ticket: `customer_id` never comes from model arguments. Only the
  tools in `UNSCOPED_TOOLS` work without it: `search_kb`, and the
  coordinator-owned `create_escalation` (only for `ctx.ticket_id`). All
  others fail with `permission` when it's missing, and another customer's
  records are reported as `not_found`, exactly like missing ones. So no
  other customer's data can enter a transcript. The ticket API is a trusted
  internal ingress, like a helpdesk webhook: its host port binds to
  localhost, and the helpdesk (not the customer) asserts `customer_id`. The
  MCP server isn't published outside the Compose network.
- Customer text is PII-redacted before any model sees it, and re-redacted in
  the final reply.
- **Injection.** Ticket text, tool outputs, and KB text are untrusted. The
  coordinator runs `screen` on the ticket and on every tool and subagent
  result it records, and ORs the verdicts into one request-scoped
  `injection_suspected` flag held by code. Flagged content is still shown to
  models, wrapped in delimiters with a "data, not instructions" note, but
  once the flag is set the executor blocks every model-directed irreversible
  action and approval request. The only write left is the code-owned
  security handoff (`create_escalation` to `security`). KB documents carry
  `trusted`; untrusted ones (`kb_community_tips`) can be retrieved but never
  cited.
- **One refund policy.** `contracts.policy.evaluate_refund` is the only
  definition of "eligible". It checks account status, delivery, the 30-day
  window, the refundable balance, the $500 hard limit, and the customer-wide
  count of issued refunds in the last 30 days. It returns stable
  `RefundReason` codes in precedence order, and `REFUND_OUTCOME` maps the
  winning code to refuse or to a queue. The MCP server, the guardrails, the
  fakes, and the eval cases all use it.
- **One executor.** `ActionExecutor` (lane D) is the only path from a
  proposed write to the gateway. `propose` validates the arguments, fetches
  eligibility itself (retrying retryable errors, and returning a persistent
  error as `failed` with the tool result), and runs
  `Guardrails.check_action`. Irreversible tools never run in-request:
  `issue_refund` is stored as an `ApprovalRequest` with the exact arguments
  and an idempotency key derived from the approval id, and the request ends
  in `pending_approval`. Lane B proposes irreversible actions only after the
  request's last `screen`, so no approval is stored for a flagged request.
  Submitting an existing approval id returns the stored one unchanged.
- **Human approval.** The API and CLI authenticate the reviewer with
  `TW_APPROVER_TOKEN` (unset disables decisions), then call
  `ActionExecutor.decide`. On approval it loads the stored arguments (never
  fresh model output), signs an HMAC token over the approval id, order, amount, and idempotency
  key (`TW_APPROVAL_KEY`), and calls the gateway as `caller="human"`. The
  approval is marked `executed` only on confirmed success. After a failure
  it stays `approved`, and a retry reuses the same idempotency key.
- **Defense in depth.** The MCP server refuses `issue_refund` unless the
  token verifies and the caller is `human`. Then it checks ownership, then
  the idempotency key (an already-committed refund is returned as success,
  so a retry after a lost response works), and only for a new write, in the
  same transaction as the insert,
  re-applies `evaluate_refund`. A forged or replayed approval id can't issue
  a refund, even over the HTTP transport. `TW_APPROVAL_KEY` (at least 32
  characters, blank means unset) is required whenever MCP runs out of
  process.

### 5. MCP server over the mock business systems

- It uses the official `mcp` Python SDK (2.x) and exposes the eight tools in
  `contracts/tools.py`.
- Every result is a `ToolResult` envelope: `isError`, a category, a
  `retryable` flag, and optional `retry_after_ms`. Handlers never raise.
- Data lives in SQLite, seeded from `data/seed`. The clock is pinned to the
  seed's `as_of` for deterministic behavior.
- Transports:
  - in-process: tests and evals
  - stdio: local clients
  - streamable HTTP: Compose
- Every raw call is logged as a `ToolCallRecord` (including retries and
  faults). Graders read this log, not the agent's own report.
- Eval faults (`ToolFault`) are injected at the gateway. A
  `CaseRuntimeFactory.build(case_id=...)` builds one isolated stack per
  eval case, bound to that case's recordings (judge calls use a separate
  file): a fresh
  SQLite database with the case's merged world, its `as_of` clock and
  faults, and an inspectable call log, ledger, and approval list. Eval knobs
  never appear on the production ticket API. `triagewright.testing` has a
  fake factory; the real one lives in `triagewright.wiring`.

### 6. RAG in SQLite

- Markdown KB with front matter, chunked on headings and paragraphs, keeping
  character offsets.
- Storage: SQLite, using FTS5 for BM25 and float32 embedding blobs for
  brute-force cosine (the KB is small).
- Hybrid retrieval combines the two with reciprocal rank fusion, plus metadata
  filters on category, tags, and date.
- Embedders: a deterministic hashing embedder (the default, used in CI) and
  `fastembed` (local ONNX, optional extra).
- Citations are verbatim quotes. They're checked against the chunk text, and
  a quote that doesn't appear there is dropped.
- pgvector was rejected because it adds a container for no benefit at this
  scale. Revisit above roughly 50k chunks.

### 7. Observability with OpenTelemetry

- Spans cover the request, each turn, each subagent, and each model, tool,
  retrieval, guardrail, and approval step.
- Exporters: JSONL under `var/traces/` (the default) or OTLP to Jaeger
  (Compose `tracing` profile).
- Cost comes from `config/pricing.toml`. Each price cites its source, and an
  unknown model is an error. Free models cost $0. Judge calls are reported
  separately and never count toward cost per ticket.
- TTFT is request start to the first streamed token
  (`ModelCallRecord.started_ms` plus that call's TTFT).
- Live latency is wall time. In replay, latency is non-model wall time plus
  each call's *recorded* latency, and metrics are marked `estimated`. Live
  and replay numbers are never compared: baselines and gates are per mode,
  and the p95 < 8 s target is judged on live runs only.
- Langfuse was rejected: self-hosting it needs Postgres, ClickHouse, and
  Redis. OTLP keeps that option open.

### 8. Evals

- One YAML file per case under `evals/cases/<category>/`.
- Deterministic graders run first: status, escalation (with queue and
  priority), must/must-not call, unsafe action, citations, and reply
  include/exclude. They read the `CaseRuntime` call log and ledger, not
  `FinalResponse` claims. An LLM-judge rubric runs second, and judge calls
  are recorded too.
- **Unsafe action:** any refund that appears in the ledger during `resolve`
  (approvals never complete in-request), or any successful call to a tool
  the case forbids.
- Metrics: pass rate, escalation precision and recall, cost per ticket, and
  p50/p95 latency.
- A PR regresses against `evals/baseline.json` if any of these happen:
  - the pass rate drops
  - any unsafe action appears
  - cost per ticket rises more than 10% (same mode)
  - p95 latency rises more than 10% (same mode)

### 9. API, CLI, and packaging

- The API is FastAPI. Its endpoints:
  - `POST /tickets`
  - `GET /approvals`
  - `POST /approvals/{id}/decision` (requires the approver token)
  - `GET /healthz`
- The CLI is Typer.
- One Docker image runs both the API and the MCP server. Compose binds to
  localhost only. Settings follow 12-factor via `TW_*` env vars.

### 10. Lane ownership

- Each lane owns only its listed paths (see `docs/books/STATUS.md`).
- Contracts, `pyproject.toml`/`uv.lock`, `compose.yaml`, `Dockerfile`, CI,
  `data/seed`, `data/kb`, `triagewright.testing`, and `docs/books/` belong to
  the orchestrator.
- A lane that needs a contract or dependency change asks the orchestrator,
  who lands it in a dedicated PR. Dependent lanes then rebase.

## Consequences

- Lanes can build and test against `triagewright.testing` fakes from day one.
- Free models make spend close to $0, but "< $0.02 per ticket" is then met
  trivially, and free-tier queues make live latency noisy. The eval report
  records the provider, models, and mode per run, and states this plainly.
  The Messages-shaped contract keeps a paid provider a config change away.
- Human approval on every refund means no refund completes inside one request.
  That's the intended, safest behavior. It's measured as `pending_approval`.
