# Status: task DAG

Owned by the orchestrator. Workers never edit `docs/books/`.

States: `todo` → `running` → `in-review` → `merged` (or `blocked`).

## Orchestrator-owned paths

`src/triagewright/contracts/**`, `src/triagewright/testing/**`, `src/triagewright/config.py`,
`src/triagewright/data.py`, `src/triagewright/wiring.py` (integration), `schemas/**`,
`scripts/export_schemas.py`, `data/seed/**`, `data/kb/**`, `docs/contracts/**`,
`docs/adr/ADR-001-*`, `docs/books/**`, `pyproject.toml`, `uv.lock`, `Dockerfile`,
`compose.yaml`, `.github/**`, `.env.example`.

## M0: Wave 0

| id | deps | owned paths | worker | state | PR |
| --- | --- | --- | --- | --- | --- |
| W0.1 bootstrap repo, CI, protection | – | (all, pre-fan-out) | orchestrator | merged (protection blocked, R3) | direct push 9255512 |
| W0.2 contracts, fakes, seed, ADR-001 | W0.1 | orchestrator paths | orchestrator | running | – |

## M1: lanes (fan out after W0.2 merges)

| id | deps | owned paths | worker | state | PR |
| --- | --- | --- | --- | --- | --- |
| A1 MCP server + mock systems | W0.2 | `src/triagewright/mcp_server/**`, `tests/mcp_server/**` | – | todo | – |
| B1 model adapters (openrouter, replay/record) | W0.2 | `src/triagewright/llm/**`, `tests/llm/**` | – | todo | – |
| B2 coordinator loop + subagents | W0.2 | `src/triagewright/agent/**`, `tests/agent/**` | – | todo | – |
| C1 RAG ingest/retrieve/citations | W0.2 | `src/triagewright/rag/**`, `tests/rag/**` | – | todo | – |
| D1 guardrails, approvals, action executor | W0.2 | `src/triagewright/guardrails/**`, `tests/guardrails/**` | – | todo | – |
| E1 tracing, pricing, cost, latency | W0.2 | `src/triagewright/observability/**`, `tests/observability/**`, `config/pricing.toml` | – | todo | – |
| F1 eval harness + graders + report | W0.2 | `src/triagewright/evals/**`, `tests/evals/**`, `evals/baseline.json`, `evals/README.md` | – | todo | – |
| G1 cases: happy path + ambiguous | W0.2 | `evals/cases/happy_path/**`, `evals/cases/ambiguous/**` | – | todo | – |
| G2 cases: policy edge + escalation | W0.2 | `evals/cases/policy_edge/**`, `evals/cases/escalation_required/**` | – | todo | – |
| G3 cases: injection + tool failure | W0.2 | `evals/cases/injection/**`, `evals/cases/tool_failure/**` | – | todo | – |
| H1 HTTP API + CLI | W0.2 | `src/triagewright/api/**`, `src/triagewright/cli/**`, `tests/api/**`, `tests/cli/**` | – | todo | – |
| I1 architecture docs (deferred to M2) | X1 | `docs/architecture.md`, `docs/adr/` (except ADR-001), `README.md` | – | todo | – |
| J1 synthetic data generator (deferred to M2, only if needed) | X1 | `src/triagewright/datagen/**`, `tests/datagen/**`, `data/generated/**` | – | todo | – |

## M1 integration

| id | deps | owned paths | worker | state | PR |
| --- | --- | --- | --- | --- | --- |
| X1 first 20 evals green (recorded once, after A, C, D land) | A1, B1, B2, C1, D1, F1, G1–G3 | `src/triagewright/wiring.py`, `evals/fixtures/recordings/**` | – | todo | – |

## M2–M3

Planned after the M1 checkpoint.
