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
| W0.2 contracts, fakes, seed, ADR-001 | W0.1 | orchestrator paths | orchestrator | merged (council: 5 rounds, all approve) | [#1](https://github.com/rick-learns/triagewright/pull/1) |
| W0.3 gitleaks on PRs | W0.1 | `.github/**` | orchestrator | merged | [#2](https://github.com/rick-learns/triagewright/pull/2) |
| W0.4 Gemini free provider, Podman | W0.2 | orchestrator paths | orchestrator | merged (sol-loop: 3 rounds, approve) | [#4](https://github.com/rick-learns/triagewright/pull/4) |

## M1: lanes (fanned out 2026-11-19, one T3 thread and worktree per lane)

| id | deps | owned paths | worker | state | PR |
| --- | --- | --- | --- | --- | --- |
| A1 MCP server + mock systems | W0.2 | `src/triagewright/mcp_server/**`, `tests/mcp_server/**` | lane thread (claude-opus-5-5) | in-review | [#13](https://github.com/rick-learns/triagewright/pull/13) |
| B1 model adapters (gemini, openrouter, replay/record) | W0.4 | `src/triagewright/llm/**`, `tests/llm/**` | lane thread (claude-opus-5-5) | in-review | [#11](https://github.com/rick-learns/triagewright/pull/11) |
| B2 coordinator loop + subagents | W0.2 | `src/triagewright/agent/**`, `tests/agent/**` | lane thread (claude-opus-5-5) | running | – |
| C1 RAG ingest/retrieve/citations | W0.2 | `src/triagewright/rag/**`, `tests/rag/**` | lane thread (claude-opus-5-5) | merged (sol-loop 2, approve) | [#9](https://github.com/rick-learns/triagewright/pull/9) |
| D1 guardrails, approvals, action executor | W0.2 | `src/triagewright/guardrails/**`, `tests/guardrails/**` | lane thread (claude-opus-5-5) | running | – |
| E1 tracing, pricing, cost, latency | W0.2 | `src/triagewright/observability/**`, `tests/observability/**`, `config/pricing.toml` | lane thread (claude-opus-5-5) | merged (sol-loop 2, approve) | [#7](https://github.com/rick-learns/triagewright/pull/7) |
| F1 eval harness + graders + report | W0.2 | `src/triagewright/evals/**`, `tests/evals/**`, `evals/baseline.json`, `evals/README.md` | lane thread (claude-opus-5-5) | in-review | [#12](https://github.com/rick-learns/triagewright/pull/12) |
| G1 cases: happy path + ambiguous | W0.2 | `evals/cases/happy_path/**`, `evals/cases/ambiguous/**` | lane thread (claude-opus-5-5) | merged (sol-loop 3, approve) | [#6](https://github.com/rick-learns/triagewright/pull/6) |
| G2 cases: policy edge + escalation | W0.2 | `evals/cases/policy_edge/**`, `evals/cases/escalation_required/**` | lane thread (claude-opus-5-5) | merged (sol-loop 2; exact-refund gap accepted, follow-up) | [#5](https://github.com/rick-learns/triagewright/pull/5) |
| G3 cases: injection + tool failure | W0.2 | `evals/cases/injection/**`, `evals/cases/tool_failure/**` | lane thread (claude-opus-5-5) | merged (sol-loop 2, approve) | [#8](https://github.com/rick-learns/triagewright/pull/8) |
| H1 HTTP API + CLI | W0.2 | `src/triagewright/api/**`, `src/triagewright/cli/**`, `tests/api/**`, `tests/cli/**` | lane thread (claude-opus-5-5) | merged (sol-loop 3; CLI-token S1 resolved by ADR amendment) | [#10](https://github.com/rick-learns/triagewright/pull/10) |
| I1 architecture docs (deferred to M2) | X1 | `docs/architecture.md`, `docs/adr/` (except ADR-001), `README.md` | – | todo | – |
| J1 synthetic data generator (deferred to M2, only if needed) | X1 | `src/triagewright/datagen/**`, `tests/datagen/**`, `data/generated/**` | – | todo | – |

## M1 integration

| id | deps | owned paths | worker | state | PR |
| --- | --- | --- | --- | --- | --- |
| X1 first 20 evals green (recorded once, after A, C, D land) | A1, B1, B2, C1, D1, F1, G1–G3 | `src/triagewright/wiring.py`, `evals/fixtures/recordings/**` | – | todo | – |

## M2–M3

Planned after the M1 checkpoint.
