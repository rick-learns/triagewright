# triagewright

A local-first AI customer-support resolution agent. A ticket comes in. The
agent triages it, calls business tools over MCP, answers from a knowledge base
with citations, and hands off to a human when policy or confidence requires it.
Irreversible actions (refunds) always wait for a human approval.

> Status: under construction (M0, contracts and skeleton). See `docs/books/STATUS.md`.

## Quick start

```bash
cp .env.example .env          # pick a provider; add its API key to .env (never commit it)
uv sync
uv run pytest                 # unit + integration tests, no network
uv run triagewright-evals smoke   # eval smoke run on recorded model calls
docker compose up             # API on http://localhost:8080, MCP server on :8765
```

## Constraints

- Local only. No cloud resources. Synthetic data only, no real PII.
- Secrets come from the environment. Only `.env.example` is committed.
- CI never calls a model: evals replay recorded fixtures.
- Model spend for the whole build is capped at $50 (see `docs/books/BUDGET.md`).

## Layout

| Path | What |
| --- | --- |
| `src/triagewright/contracts/` | Shared Pydantic contracts and ports (orchestrator-owned) |
| `src/triagewright/mcp_server/` | MCP server over the mock business systems |
| `src/triagewright/agent/`, `llm/` | Coordinator loop, subagents, model adapters |
| `src/triagewright/rag/` | KB ingest, chunking, embeddings, hybrid retrieval, citations |
| `src/triagewright/guardrails/` | PII redaction, refund policy, injection screening, approvals |
| `src/triagewright/observability/` | Tracing, token cost, TTFT, latency |
| `src/triagewright/evals/`, `evals/` | Eval harness, graders, cases, report |
| `src/triagewright/api/`, `cli/` | HTTP API and CLI |
| `src/triagewright/datagen/` | Synthetic data generator |
| `schemas/` | JSON Schemas exported from the contracts |
| `docs/` | Architecture, ADRs, and the project books |
