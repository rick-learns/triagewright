# ADR-001: Stack and architecture

- Status: Proposed (pending council review)
- Date: 2026-11-19

## Context

triagewright is a local-first customer-support agent: triage, tool use over MCP,
grounded answers with citations, and escalation to a human. It must run on one
machine with `docker compose up`, cost under $50 in model spend to build, and
be testable in CI with no API keys. Ten lanes are built in parallel, so the
contracts between them must be fixed before fan-out.

## Decisions

### 1. Python 3.12, uv, Pydantic v2 contracts as the single source of truth

All cross-lane types live in `src/triagewright/contracts/`. They're frozen,
reject unknown fields, and use integer cents. JSON Schemas in `schemas/` are
generated from them (`scripts/export_schemas.py`), and a test fails if they're
stale. Lanes talk only through the Protocols in `contracts/ports.py`.

### 2. Provider-neutral model layer shaped like the Anthropic Messages API

`ModelRequest`/`ModelResponse` mirror Messages: content blocks (`text`,
`tool_use`, `tool_result`), `stop_reason`, and usage including cache tokens.
Adapters: `anthropic` (official SDK), `openrouter`, and `replay`. The replay
adapter serves `Recording`s keyed by `ModelRequest.fingerprint()`. The default
provider is `replay`, so nothing calls a paid API unless `.env` opts in. CI
always replays. A missing recording is a hard error, never a live fallback.

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
  final answer is a forced `finalize` tool call whose input is validated
  against the `FinalResponse` fields (structured output through tool schema).
  It's validated with Pydantic and retried once with the validation error.
- **Subagents** (`triage`, `answer`, `action`, `escalation`) get only their
  typed input (`TriageInput`, `AnswerInput`, …), never the coordinator
  transcript. Each returns its typed output, validated before the coordinator
  sees it. Each subagent may run its own bounded tool loop. The answer
  subagent may use `search_kb` only. The action subagent may use only the
  tools in `ActionInput.allowed_tools`.

### 4. Guardrails are code, not prompts

- Customer text is PII-redacted before any model sees it, and re-redacted in
  the final reply.
- Tool outputs and KB text count as untrusted data. The `screen` check flags
  instruction-like content, and the agent wraps that content in delimiters
  with an explicit "data, not instructions" note.
- Every proposed write passes `Guardrails.check_action` inside a guarded tool
  gateway before it reaches MCP.
- Irreversible tools (`issue_refund`) always return `require_approval`. An
  `ApprovalRequest` is stored and the request ends in `pending_approval`. A
  human decides through the API or CLI, and only then does the stored action
  run, with its `approval_id`.
- Defense in depth: the MCP server rejects `issue_refund` without an
  `approval_id`, enforces the refund window and the refundable balance, and
  dedupes by idempotency key.

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
- Eval faults (`ToolFault`) are injected at the gateway.

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
- Exporters: JSONL under `var/traces/` (the default), console, or OTLP to
  Jaeger (Compose `tracing` profile).
- Cost comes from `config/pricing.toml`. Each price cites its source, and an
  unknown model is an error. TTFT comes from streaming.
- Langfuse was rejected: self-hosting it needs Postgres, ClickHouse, and
  Redis. OTLP keeps that option open.

### 8. Evals

- One YAML file per case under `evals/cases/<category>/`.
- Deterministic graders run first: status, escalation, must/must-not call,
  unsafe action, citations, and reply include/exclude. An LLM-judge rubric
  runs second, and judge calls are recorded too.
- Metrics: pass rate, escalation precision and recall, cost per ticket, and
  p50/p95 latency.
- A PR regresses against `evals/baseline.json` if any of these happen:
  - the pass rate drops
  - any unsafe action appears
  - cost per ticket rises more than 10%
  - p95 latency rises more than 10%

### 9. API, CLI, and packaging

- The API is FastAPI. Its endpoints:
  - `POST /tickets`
  - `GET /approvals`
  - `POST /approvals/{id}/decision`
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
- A provider swap is a config change. But metrics from free models (cost $0,
  slower queues) aren't comparable with Claude metrics, so the eval report
  records the provider and models per run.
- Human approval on every refund means no refund completes inside one request.
  That's the intended, safest behavior. It's measured as `pending_approval`.
