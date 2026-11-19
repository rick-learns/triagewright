# Status: task DAG

Owned by the orchestrator. Workers never edit `docs/books/`.

States: `todo` → `running` → `in-review` → `merged` (or `blocked`).

## Orchestrator-owned paths

`src/triagewright/contracts/**`, `src/triagewright/testing/**`, `src/triagewright/config.py`,
`src/triagewright/data.py`, `src/triagewright/wiring.py` (integration), `schemas/**`,
`scripts/export_schemas.py`, `data/seed/**`, `data/kb/**`, `docs/contracts/**`,
`docs/adr/ADR-001-*`, `docs/books/**`, `pyproject.toml`, `uv.lock`, `Dockerfile`,
`compose.yaml`, `.github/**`, `.env.example`.

Handed to M2/M3 lanes until their PRs merge (one writer per path): `Dockerfile`,
`compose.yaml`, `.env.example` to P1; `src/triagewright/testing/**` to RF2;
`src/triagewright/wiring.py` (and its new `src/triagewright/wiring/**` package) to RF3.

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
| A1 MCP server + mock systems | W0.2 | `src/triagewright/mcp_server/**`, `tests/mcp_server/**` | lane thread (claude-opus-5-5) | merged (sol-loop 3, changes: last should fixed after review, not re-reviewed) | [#13](https://github.com/rick-learns/triagewright/pull/13) |
| B1 model adapters (gemini, openrouter, replay/record) | W0.4 | `src/triagewright/llm/**`, `tests/llm/**` | lane thread (claude-opus-5-5) | merged (sol-loop 3) | [#11](https://github.com/rick-learns/triagewright/pull/11) |
| B2 coordinator loop + subagents | W0.2 | `src/triagewright/agent/**`, `tests/agent/**` | lane thread (claude-opus-5-5) | merged (sol-loop 5) | [#14](https://github.com/rick-learns/triagewright/pull/14) |
| C1 RAG ingest/retrieve/citations | W0.2 | `src/triagewright/rag/**`, `tests/rag/**` | lane thread (claude-opus-5-5) | merged (sol-loop 2, approve) | [#9](https://github.com/rick-learns/triagewright/pull/9) |
| D1 guardrails, approvals, action executor | W0.2 | `src/triagewright/guardrails/**`, `tests/guardrails/**` | lane thread (claude-opus-5-5) | merged (sol-loop 5) | [#16](https://github.com/rick-learns/triagewright/pull/16) |
| E1 tracing, pricing, cost, latency | W0.2 | `src/triagewright/observability/**`, `tests/observability/**`, `config/pricing.toml` | lane thread (claude-opus-5-5) | merged (sol-loop 2, approve) | [#7](https://github.com/rick-learns/triagewright/pull/7) |
| F1 eval harness + graders + report | W0.2 | `src/triagewright/evals/**`, `tests/evals/**`, `evals/baseline.json`, `evals/README.md` | lane thread (claude-opus-5-5) | merged (sol-loop 3) | [#12](https://github.com/rick-learns/triagewright/pull/12) |
| G1 cases: happy path + ambiguous | W0.2 | `evals/cases/happy_path/**`, `evals/cases/ambiguous/**` | lane thread (claude-opus-5-5) | merged (sol-loop 3, approve) | [#6](https://github.com/rick-learns/triagewright/pull/6) |
| G2 cases: policy edge + escalation | W0.2 | `evals/cases/policy_edge/**`, `evals/cases/escalation_required/**` | lane thread (claude-opus-5-5) | merged (sol-loop 2; exact-refund gap accepted, follow-up) | [#5](https://github.com/rick-learns/triagewright/pull/5) |
| G3 cases: injection + tool failure | W0.2 | `evals/cases/injection/**`, `evals/cases/tool_failure/**` | lane thread (claude-opus-5-5) | merged (sol-loop 2, approve) | [#8](https://github.com/rick-learns/triagewright/pull/8) |
| H1 HTTP API + CLI | W0.2 | `src/triagewright/api/**`, `src/triagewright/cli/**`, `tests/api/**`, `tests/cli/**` | lane thread (claude-opus-5-5) | merged (sol-loop 3; CLI-token S1 resolved by ADR amendment) | [#10](https://github.com/rick-learns/triagewright/pull/10) |
| I1 architecture docs (deferred to M2) | X1 | `docs/architecture.md`, `docs/adr/` (except ADR-001), `README.md` | – | moved to M2 (below) | – |
| J1 synthetic data generator (deferred to M2, only if needed) | X1 | `src/triagewright/datagen/**`, `tests/datagen/**`, `data/generated/**` | – | moved to M2 (below) | – |

## M1 integration

| id | deps | owned paths | worker | state | PR |
| --- | --- | --- | --- | --- | --- |
| X1 compose the real lanes for API, CLI, evals | A1, B1, B2, C1, D1, E1, F1, G1–G3, H1 | `src/triagewright/wiring.py` | orchestrator | merged (sol-loop) | [#17](https://github.com/rick-learns/triagewright/pull/17) |
| F1b `min_refund_cents` expectation | F1, G2 | `src/triagewright/evals/**`, `tests/evals/**`, `evals/cases/policy_edge/policy-002.yaml`, `policy-006.yaml` | orchestrator | merged | [#15](https://github.com/rick-learns/triagewright/pull/15) |

## M2–M3: one writer for the agent core, lanes for everything else

The agent core (`src/triagewright/agent/**`, `src/triagewright/llm/**`, `tests/agent/**`,
`tests/llm/**`) has one writer, the orchestrator: live-quality fixes and the latency work
both change routing and the model-call sequence, and live evals share one free-tier
quota, so they run serially. Lanes never make live model calls (fakes, `--stub`, replay).
Lanes launched 2026-11-19 09:45 UTC (7 at once), one T3 thread and worktree per lane from `origin/main`.

### Orchestrator (serial)

| id | deps | owned paths | worker | state | PR |
| --- | --- | --- | --- | --- | --- |
| Q1 pace live calls to the free-tier quota | X1 | `src/triagewright/llm/**`, `src/triagewright/evals/runner.py`, `src/triagewright/config.py` | orchestrator | merged (sol-loop 3) | [#18](https://github.com/rick-learns/triagewright/pull/18) |
| Q2 code decides every refund outcome | Q1 | agent core | orchestrator | merged (sol-loop 5) | [#19](https://github.com/rick-learns/triagewright/pull/19) |
| Q3 pending refunds stay with the approver; trace model errors | Q2 | agent core, `src/triagewright/observability/**` | orchestrator | merged (sol-loop 3) | [#20](https://github.com/rick-learns/triagewright/pull/20) |
| Q4 reviewer reasons code already handled | Q3 | agent core, `docs/contracts/behavior.md` | orchestrator | merged (sol-loop 2) | [#21](https://github.com/rick-learns/triagewright/pull/21) |
| Q5 remaining live failures (escalation queues, over-cap proposals, tool failures, 503 retries) | Q4 | agent core | orchestrator | todo | – |
| L1 latency: fewer sequential model calls per ticket (p95 < 8 s) | Q4 | agent core | orchestrator | running (refund settle in code, sol-loop) | – |
| X2 record replay fixtures, full live eval, `evals/report.md` | Q5, L1, F2 | `evals/fixtures/recordings/**`, `evals/report.md` (new), `evals/baseline.json` | orchestrator (live, serial) | todo | – |

### Lanes (parallel, disjoint owned paths)

| id | deps | owned paths | worker | state | PR |
| --- | --- | --- | --- | --- | --- |
| I1 architecture doc, ADRs, exam map | X1 | `docs/architecture.md` (new), `docs/adr/ADR-002-*` and later (new; not ADR-001), `docs/exam-map.md` (new) | lane thread (claude-opus-5-5) | running | – |
| R1 README rewrite | X1 | `README.md`, `docs/assets/**` (new) | lane thread (claude-opus-5-5) | in-review | [#27](https://github.com/rick-learns/triagewright/pull/27) |
| J1 synthetic data generator | X1 | `src/triagewright/datagen/**`, `tests/datagen/**`, `data/generated/**` (new) | lane thread (claude-opus-5-5) | merged | [#25](https://github.com/rick-learns/triagewright/pull/25) |
| F2 eval report generator + case review checklist | F1 | `src/triagewright/evals/**` (except `runner.py`), `tests/evals/**` (except `test_runner.py`), `evals/README.md`, `evals/REVIEWING.md` (new) | lane thread (claude-opus-5-5) | merged | [#26](https://github.com/rick-learns/triagewright/pull/26) |
| P1 fresh-clone `podman compose up` | X1 | `Dockerfile`, `compose.yaml`, `.env.example`, `.dockerignore`, `scripts/compose_smoke.sh` (new) | lane thread (claude-opus-5-5) | running | – |
| RF2 split the test fakes | X1 | `src/triagewright/testing/**`, `tests/contracts/test_fakes.py`, `tests/contracts/test_fakes_exports.py` (new) | lane thread (claude-opus-5-5) | merged | [#23](https://github.com/rick-learns/triagewright/pull/23) |
| RF3 split the wiring module | X1 | `src/triagewright/wiring.py` (removed), `src/triagewright/wiring/**` (new package), `tests/test_wiring.py` | lane thread (claude-opus-5-5) | merged | [#24](https://github.com/rick-learns/triagewright/pull/24) |

### Later

| id | deps | owned paths | worker | state | PR |
| --- | --- | --- | --- | --- | --- |
| RF1 split the agent core (`coordinator.py`, `runtime.py`, `tests/agent/test_coordinator.py`) and `llm/openai_compat.py` | Q5, L1 | agent core | lane thread | todo | – |
| D2 demo script and recording on replay | X2, R1 | `scripts/demo/**` (new), `docs/demo.md` (new) | lane thread | todo | – |
| C2 council review of the PoC before the M3 checkpoint | X2, I1, R1, P1 | read-only | council | todo | – |

Human step: flipping eval cases to `reviewed: true` is the owner's (or a named reviewer's), using `evals/REVIEWING.md` from F2.
