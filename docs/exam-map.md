# Exam map: Claude Certified Architect – Foundations

This maps the exam's domains to the places in this repository that demonstrate
them. Paths are relative to the repository root. The design is described in
[architecture.md](architecture.md).

## Source

The domain list and task statements come from the official exam guide:
*Claude Certified Architect – Foundations Exam Guide*, version 1.0, effective
July 2026, exam code CCAR-F. It is linked from the public certification page in
Anthropic Partner Academy,
<https://anthropic-partners.skilljar.com/claude-certified-architect-foundations-certification>
(checked 2026-11-19). The Pearson VUE program page,
<https://www.pearsonvue.com/us/en/anthropic.html>, lists the exams that get updated
content on 2026-12-07. CCAR-F isn't one of them.

The guide has five weighted domains:

| # | Domain | Weight |
| --- | --- | --- |
| 1 | Agentic Architecture & Orchestration | 27% |
| 2 | Tool Design & MCP Integration | 18% |
| 3 | Claude Code Configuration & Workflows | 20% |
| 4 | Prompt Engineering & Structured Output | 20% |
| 5 | Context Management & Reliability | 15% |

Guardrails and human-in-the-loop, and evaluation and observability, are not
domains of their own in the guide. They fall under task statements in domains
1, 4, and 5, so they get their own sections below with those statements named.
The guide's Scenario 1, a customer support resolution agent with MCP tools for
customer lookup, orders, refunds, and escalation, is close to what this
repository builds.

## 1. Agentic Architecture & Orchestration

| Task statement | Where | Notes |
| --- | --- | --- |
| 1.1 Agentic loops | `src/triagewright/agent/runtime.py` (`RequestRun.run_loop`) | Explicit loop on `stop_reason`: `tool_use` runs tools and appends `tool_result` blocks, `end_turn` forces the output tool, `max_tokens` gets one continuation, `pause_turn` continues, `refusal` stops. Bounded by `max_turns`, per-turn and total timeouts. |
| 1.2 Coordinator and subagents | `src/triagewright/agent/coordinator.py`, `src/triagewright/agent/subagents.py` | Hub and spoke: the coordinator calls `answer` and `action` as tools; code always runs triage first and the escalation reviewer last, so the model can't skip them. |
| 1.3 Subagent context passing | `src/triagewright/contracts/agents.py` | Each subagent gets only its typed input (`TriageInput`, `AnswerInput`, `ActionInput`, `EscalationInput`), never the coordinator transcript, and returns a validated typed output. |
| 1.4 Enforcement and handoff | `src/triagewright/agent/coordinator.py` (`_check_final`, `_summary`), `docs/adr/ADR-004-code-decides-refund-outcomes.md` | A programmatic gate, not a prompt rule, requires a refund proposal and a policy citation before a refund reply is accepted. Handoff summaries state goal, facts checked, what was attempted, and why. |
| 1.5 Tool call interception | `src/triagewright/agent/runtime.py` (`call_tool`, `defer`), `src/triagewright/guardrails/executor.py` | No Agent SDK hooks; the same idea in plain code. Every tool call passes one wrapper that sets a trusted `CallContext`, retries, and screens the result; irreversible proposals are intercepted and deferred to the executor. |
| 1.6 Task decomposition | `src/triagewright/agent/prompts.py` (`COORDINATOR`) | Fixed split by concern: triage, grounded answer, records and actions, review. |
| 1.7 Session state and resumption | `src/triagewright/guardrails/approvals.py`, `src/triagewright/evals/runner.py` | Not a conversation session. Durable state is the approval: a repeat approve resumes execution with the same idempotency key. The eval runner resumes from its results file (`--resume`). |

## 2. Tool Design & MCP Integration

| Task statement | Where | Notes |
| --- | --- | --- |
| 2.1 Tool interfaces | `src/triagewright/contracts/tools.py` (`TOOL_SPECS`) | Eight tools, each with a description, side-effect class, and Pydantic input and output models. `search_kb`'s `categories` field documents the opt-in for untrusted pages. |
| 2.2 Structured error responses | `src/triagewright/contracts/tools.py` (`ToolResult`, `ToolError`, `ErrorCategory`), `src/triagewright/mcp_server/server.py` | Every result is an envelope with `is_error`, a category, `retryable`, and `retry_after_ms`. Handlers never raise; the MCP `CallToolResult` carries the same JSON as `structuredContent`. Access failures (`transient`, `timeout`) are distinct from `not_found` and from valid empty results. |
| 2.3 Tool distribution and tool choice | `src/triagewright/agent/coordinator.py` (`READ_TOOLS`, `REFUND_READ_TOOLS`), `src/triagewright/agent/subagents.py` (`ACTION_TOOLS`) | Each agent gets only its tools: the answer subagent only `search_kb`, the action subagent read tools plus `propose_action`. `create_escalation` is code-owned (`CODE_OWNED_TOOLS`). Output tools are forced with `ToolChoice(type="tool")`. |
| 2.4 MCP servers in agent workflows | `src/triagewright/mcp_server/`, `src/triagewright/wiring.py` (`BackgroundMcpGateway`), `compose.yaml` | A real MCP server on the official SDK over stdio or streamable HTTP, with trusted call context in `_meta`. In-process for tests and evals, HTTP in Compose. |
| 2.5 Built-in tools | – | Not demonstrated: the agent doesn't use Claude Code's built-in tools. |

## 3. Claude Code Configuration & Workflows

| Task statement | Where | Notes |
| --- | --- | --- |
| 3.1–3.5 CLAUDE.md, commands and skills, path rules, plan mode, iterative refinement | – | Not demonstrated in the repository: it has no `CLAUDE.md`, commands, or skills. |
| 3.6 CI/CD | `.github/workflows/ci.yml` | CI runs lint, type checks, tests, eval smoke, and secret scanning with `TW_LLM_PROVIDER=replay` and no keys. Claude Code is not part of CI. |

## 4. Prompt Engineering & Structured Output

| Task statement | Where | Notes |
| --- | --- | --- |
| 4.1 Explicit criteria | `src/triagewright/agent/prompts.py` | Each prompt lists explicit criteria, such as when `needs_human` applies, what each status means, and what never to ask for. |
| 4.2 Few-shot prompting | – | Not used. |
| 4.3 Structured output with tool use and JSON Schema | `src/triagewright/agent/runtime.py` (`run_loop`), `src/triagewright/contracts/`, `schemas/`, `scripts/export_schemas.py` | Results arrive as a forced tool call whose input schema is the Pydantic model's JSON Schema (`finalize`, `submit_<name>`). Contracts are exported to `schemas/`, and a check fails if they're stale. |
| 4.4 Validation, retry, and feedback | `src/triagewright/agent/runtime.py` (`validation_message`), `src/triagewright/agent/coordinator.py` (`_check_final`) | Invalid output is sent back with the validation errors for one repair; a code check that rejects output says why. A second failure is the safe fallback. |
| 4.5 Batch processing | – | Not used. The eval runner bounds concurrency and throttles case starts instead. |
| 4.6 Multi-pass review | `src/triagewright/agent/subagents.py` (`review`), `src/triagewright/evals/judge.py` | An independent escalation reviewer sees the draft with fresh context. The eval judge grades a rubric only after the deterministic graders. |

## 5. Context Management & Reliability

| Task statement | Where | Notes |
| --- | --- | --- |
| 5.1 Preserving critical information | `src/triagewright/agent/runtime.py` (`RequestRun`) | Facts code verified (tool results, KB hits, outcomes) are kept in request state and passed to subagents as `facts`, rather than relying on the transcript. |
| 5.2 Escalation and ambiguity | `docs/contracts/behavior.md`, `src/triagewright/agent/coordinator.py` (`_route`, `_order_unclear`) | Explicit triggers: customer asks for a human, safety, billing disputes, policy limits, tool failure. When the orders don't narrow to one, the agent asks which order instead of guessing. |
| 5.3 Error propagation | `src/triagewright/agent/runtime.py` (`call_tool`), `src/triagewright/llm/errors.py` | Retryable tool errors are retried locally; persistent ones are recorded and routed to tier1 with `tool_failure`. Typed model errors carry the status. |
| 5.5 Human review | `src/triagewright/guardrails/executor.py`, `src/triagewright/api/app.py` | Every refund waits for an authenticated human decision. See also the next section. |
| 5.6 Provenance | `src/triagewright/agent/runtime.py` (`cite`), `src/triagewright/rag/citations.py`, `src/triagewright/rag/retriever.py` | RAG with hybrid BM25 and vector ranking. A citation needs a trusted chunk seen in this request and a verbatim quote; untrusted pages are never cited. |
| Reliability (pacing, fallbacks) | `src/triagewright/llm/ratelimit.py`, `src/triagewright/pacing.py`, `src/triagewright/llm/openai_compat.py`, `docs/adr/ADR-002-gemini-free-tier-default-provider.md`, `docs/adr/ADR-003-pace-live-calls-per-quota.md` | Free-tier pacing per quota and model, waits kept off deadlines, model retries with `Retry-After`, a safe fallback on every bound, and a free provider fallback by config. |
| 5.4 Large codebase exploration | – | Not applicable. |

## Guardrails and human-in-the-loop

Related task statements: 1.4, 1.5, 5.2, and 5.5.

| Topic | Where | Notes |
| --- | --- | --- |
| PII redaction | `src/triagewright/guardrails/redaction.py` | Numbered placeholders before any model call, and again on the reply. |
| Injection screening | `src/triagewright/guardrails/screening.py`, `src/triagewright/agent/runtime.py` (`screen`, `as_data`) | Every untrusted input is screened; one request-scoped flag held by code blocks irreversible actions and routes writes and handoffs to security. |
| Refund policy | `src/triagewright/contracts/policy.py` (`evaluate_refund`, `REFUND_OUTCOME`) | One policy, used by the server, guardrails, fakes, and evals. |
| Human approval | `src/triagewright/guardrails/executor.py`, `src/triagewright/guardrails/approvals.py`, `src/triagewright/api/app.py`, `src/triagewright/mcp_server/service.py` | Stored arguments, an approver token on the API and CLI, an HMAC token checked by the MCP server, and idempotent execution. |
| Code-owned decisions | `docs/adr/ADR-004-code-decides-refund-outcomes.md`, `docs/adr/ADR-005-approval-queue-is-the-human-step.md` | Refund outcomes, injection routing, and escalation routing are code's; reviewer escalations code already handled are left to code. |

## Evaluation and observability

Related task statements: 4.4, 4.6, and 5.5.

| Topic | Where | Notes |
| --- | --- | --- |
| Eval cases | `evals/cases/`, `src/triagewright/contracts/evals.py` | 54 cases in six categories, including injection and tool-failure cases with injected faults. |
| Evidence-based grading | `src/triagewright/evals/graders.py`, `src/triagewright/wiring.py` (`EvalStackFactory`) | Graders read the gateway call log and ledger from an isolated per-case stack, not the agent's claims. |
| Release gate | `src/triagewright/evals/report.py` (`regressions`), `evals/baseline.json` | Unsafe actions always fail; pass rate, cost, and p95 latency compare with the same mode's baseline. |
| Tracing and metrics | `src/triagewright/observability/telemetry.py`, `src/triagewright/contracts/telemetry.py` | OpenTelemetry spans per request, turn, subagent, model, tool, and guardrail step; tokens, cost, TTFT, latency, and pacing per request. |
| Safe error spans | `docs/adr/ADR-006-span-errors-without-exception-messages.md` | Error type and HTTP status only, never exception messages. |
