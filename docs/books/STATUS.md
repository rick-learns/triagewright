# Status: task DAG

Owned by the orchestrator. Workers never edit `docs/books/`.

States: `todo` → `running` → `in-review` → `merged` (or `blocked`).

## Orchestrator-owned paths

`src/triagewright/contracts/**`, `src/triagewright/testing/**`, `src/triagewright/config.py`,
`src/triagewright/data.py`, `src/triagewright/wiring/**` (integration), `schemas/**`,
`scripts/export_schemas.py`, `data/seed/**`, `data/kb/**`, `docs/contracts/**`,
`docs/adr/ADR-001-*`, `docs/books/**`, `pyproject.toml`, `uv.lock`, `Dockerfile`,
`compose.yaml`, `.github/**`, `.env.example`.

Handed to M2/M3 lanes until their PRs merge (one writer per path). Returned: `Dockerfile`,
`compose.yaml`, `.env.example` from P1 (#29); `src/triagewright/testing/**` from RF2 (#23);
`src/triagewright/wiring/**` from RF3 (#24); `src/triagewright/llm/**` and `tests/llm/**` from
RF4 (#34); `src/triagewright/agent/**` and `tests/agent/**` from RF1 (#35). None are out now.

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
`tests/llm/**`) has one writer at a time, normally the orchestrator (while RF1 and RF4 run,
those lanes hold `agent/**` + `tests/agent/**` and `llm/**` + `tests/llm/**`): live-quality fixes and the latency work
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
| Q5 live routing fixes: planted injections to security, locked suspended accounts to tier2, asctime Retry-After | Q4, RF1, RF4 | agent core, `docs/contracts/behavior.md` | orchestrator | merged (sol-loop 5, extended: injection and account routing) | [#37](https://github.com/rick-learns/triagewright/pull/37) |
| L1 latency: fewer sequential model calls per ticket (p95 < 8 s) | Q4 | agent core, `docs/contracts/behavior.md` | orchestrator | merged (sol-loop 5; live latency not measured yet) | [#30](https://github.com/rick-learns/triagewright/pull/30) |
| X2 full live eval, `evals/report.md` | Q5, L1, F2 | `evals/report.md` (new), `evals/baseline.json` | orchestrator (live, serial) | running (Gemini free tier degraded; run on OpenRouter `:free`, labeled) | – |

### Lanes (parallel, disjoint owned paths)

| id | deps | owned paths | worker | state | PR |
| --- | --- | --- | --- | --- | --- |
| I1 architecture doc, ADRs, exam map | X1 | `docs/architecture.md` (new), `docs/adr/ADR-002-*` and later (new; not ADR-001), `docs/exam-map.md` (new) | lane thread (claude-opus-5-5) | merged | [#28](https://github.com/rick-learns/triagewright/pull/28) |
| R1 README rewrite | X1 | `README.md`, `docs/assets/**` (new) | lane thread (claude-opus-5-5) | merged | [#27](https://github.com/rick-learns/triagewright/pull/27) |
| J1 synthetic data generator | X1 | `src/triagewright/datagen/**`, `tests/datagen/**`, `data/generated/**` (new) | lane thread (claude-opus-5-5) | merged | [#25](https://github.com/rick-learns/triagewright/pull/25) |
| F2 eval report generator + case review checklist | F1 | `src/triagewright/evals/**` (except `runner.py`), `tests/evals/**` (except `test_runner.py`), `evals/README.md`, `evals/REVIEWING.md` (new) | lane thread (claude-opus-5-5) | merged | [#26](https://github.com/rick-learns/triagewright/pull/26) |
| P1 fresh-clone `podman compose up` | X1 | `Dockerfile`, `compose.yaml`, `.env.example`, `.dockerignore`, `scripts/compose_smoke.sh` (new) | lane thread (claude-opus-5-5) | merged | [#29](https://github.com/rick-learns/triagewright/pull/29) |
| RF2 split the test fakes | X1 | `src/triagewright/testing/**`, `tests/contracts/test_fakes.py`, `tests/contracts/test_fakes_exports.py` (new) | lane thread (claude-opus-5-5) | merged | [#23](https://github.com/rick-learns/triagewright/pull/23) |
| RF3 split the wiring module | X1 | `src/triagewright/wiring.py` (removed), `src/triagewright/wiring/**` (new package), `tests/test_wiring.py` | lane thread (claude-opus-5-5) | merged | [#24](https://github.com/rick-learns/triagewright/pull/24) |

Second wave (refactors; RF4–RF6 launched 10:31 UTC, RF1 at 10:38 UTC). RF1 moved ahead of Q5 because the live
fixes wait for the free-tier quota to reset; `src/triagewright/agent/**` and `tests/agent/**`
are handed to RF1 until it merges (`llm/**` and `tests/llm/**` are RF4's).

| id | deps | owned paths | worker | state | PR |
| --- | --- | --- | --- | --- | --- |
| RF1 split the agent core | L1 | `src/triagewright/agent/**`, `tests/agent/**` (handed off) | lane thread (claude-opus-5-5) | merged (sol-loop 1) | [#35](https://github.com/rick-learns/triagewright/pull/35) |
| RF4 split the model adapter | X1 | `src/triagewright/llm/**`, `tests/llm/**` (handed off) | lane thread (claude-opus-5-5) | merged (sol-loop 3) | [#34](https://github.com/rick-learns/triagewright/pull/34) |
| RF5 split the eval report and data generator | F2, J1 | `src/triagewright/evals/report.py` (and a new `src/triagewright/evals/report/` package), `tests/evals/test_report*.py`, `src/triagewright/datagen/**`, `tests/datagen/**` | lane thread (claude-opus-5-5) | merged (sol-loop 1) | [#33](https://github.com/rick-learns/triagewright/pull/33) |
| RF6 split oversized test modules | RF2, RF3 | `tests/test_wiring.py` (to `tests/wiring/**`), `tests/contracts/test_fakes*.py`, `tests/guardrails/test_executor*.py`, and new helpers in those directories (`tests/contracts/conftest.py`, `tests/contracts/fakes_helpers.py`, `tests/guardrails/conftest.py`, `tests/guardrails/executor_helpers.py`) | lane thread (claude-opus-5-5) | merged (sol-loop 1) | [#32](https://github.com/rick-learns/triagewright/pull/32) |

Third wave (launched about 14:30 UTC; all merged by 16:04 UTC).

| id | deps | owned paths | worker | state | PR |
| --- | --- | --- | --- | --- | --- |
| P2 price `gemini-3.1-flash-lite` at $0 | – | `config/pricing.toml` | orchestrator | merged (sol-loop 1) | [#36](https://github.com/rick-learns/triagewright/pull/36) |
| D3 doc pointers to the refactored modules, offline demo (replaces D2) | RF1–RF6 | `docs/exam-map.md`, `docs/architecture.md`, `docs/adr/ADR-002..007`, `docs/demo.md` (new), `scripts/demo.sh` (new), `docs/assets/demo_run.py` (new) | lane thread (claude-opus-5-5) | merged (sol-loop 3) | [#38](https://github.com/rick-learns/triagewright/pull/38) |
| T2 harden weak tests, de-flake three, report wording, refund-floor cleanup | RF5, RF6 | `tests/contracts/test_fakes_ports.py`, `tests/wiring/**`, `tests/evals/test_runner.py`, `evals/report/markdown.py`, `tests/evals/test_report_render.py`, `datagen/refunds.py`, `tests/datagen/**` | lane thread (claude-opus-5-5) | merged (sol-loop 3) | [#40](https://github.com/rick-learns/triagewright/pull/40) |
| C3 council review of the 54 eval cases | F2, G1–G3 | `evals/cases/**`, `evals/REVIEWING.md`, `evals/README.md` | council (3 rounds) + orchestrator | merged (sol-loop 1) | [#39](https://github.com/rick-learns/triagewright/pull/39) |

### Later

| id | deps | owned paths | worker | state | PR |
| --- | --- | --- | --- | --- | --- |
| C2 council review of the PoC before the M3 checkpoint | X2, I1, R1, P1 | read-only | council | todo | – |

Eval-case review: by owner instruction (14:20 UTC), a three-model council (GPT Sol, GPT Astra, Claude Opus) reviewed all 54 cases against `evals/REVIEWING.md`, and #39 set `reviewed: true` on all of them. It's recorded as a council review, not a human review.
