"""The four isolated subagents.

Each one gets only its typed input contract (as data in its first message, never
the coordinator's transcript), only its own tools, and returns its typed output,
validated by :meth:`RequestRun.run_loop`. Its output is screened like any other
untrusted content before the coordinator records it.
"""

from __future__ import annotations

from typing import cast

from pydantic import Field, ValidationError

from triagewright.agent import prompts
from triagewright.agent.runtime import LocalTool, RequestRun, validation_message
from triagewright.contracts.agents import (
    ActionInput,
    ActionResult,
    AnswerDraft,
    AnswerInput,
    EscalationDecision,
    EscalationInput,
    ProposedAction,
    SubagentName,
    TriageInput,
    TriageResult,
)
from triagewright.contracts.common import Contract, JsonObject, canonical_json
from triagewright.contracts.guardrails import ContentSource
from triagewright.contracts.llm import ToolDefinition
from triagewright.contracts.telemetry import SpanKind
from triagewright.contracts.tools import (
    IRREVERSIBLE_TOOLS,
    ToolName,
    proposal_model,
)

# Read-only tools the action subagent may call directly. Writes go through ``propose_action``.
# Not ``check_refund_eligibility``: the executor looks it up for every refund proposal, and a
# model's own lookup only repeats it (and read alone, tempts the model to settle the refund).
ACTION_TOOLS: tuple[ToolName, ...] = (
    ToolName.GET_CUSTOMER,
    ToolName.GET_ACCOUNT,
    ToolName.GET_ORDER,
    ToolName.LIST_ORDERS,
)


class ActionSummary(Contract):
    """What the action model returns. Code builds :class:`ActionResult` from records."""

    summary: str = Field(max_length=1000)


async def _subagent[T: Contract](
    run: RequestRun,
    name: SubagentName,
    system: str,
    payload: Contract,
    output: type[T],
    tools: list[LocalTool],
) -> tuple[T, bool]:
    with run.telemetry.span(f"subagent.{name.value}", SpanKind.SUBAGENT) as span:
        result = await run.run_loop(
            name.value,
            system,
            run.data_prompt(payload),
            tools,
            output,
            f"submit_{name.value}",
            prompts.submit_description(name.value),
        )
        text = canonical_json(result.model_dump(mode="json"))
        flagged = run.screen(text, ContentSource.MODEL_OUTPUT)
        span.set_attribute("injection_suspected", flagged)
    return result, flagged


async def triage(run: RequestRun) -> TriageResult:
    result, _ = await _subagent(
        run, SubagentName.TRIAGE, prompts.TRIAGE, TriageInput(ticket=run.ticket), TriageResult, []
    )
    return result


async def answer(run: RequestRun) -> tuple[AnswerDraft, bool]:
    """Search the KB and draft a grounded answer. Citations are re-checked by code."""
    payload = AnswerInput(ticket=run.ticket, triage=run.models_triage(), facts=list(run.facts))
    tools = [run.business_tool(ToolName.SEARCH_KB, "answer")]
    draft, flagged = await _subagent(
        run, SubagentName.ANSWER, prompts.ANSWER, payload, AnswerDraft, tools
    )
    citations = run.cite_all([(c.chunk_id, c.quote) for c in draft.citations])
    draft = draft.model_copy(update={"citations": citations})
    run.answer = draft
    return draft, flagged


async def action(run: RequestRun) -> tuple[ActionResult, bool]:
    """Look up records and propose writes. Irreversible proposals are deferred."""
    payload = ActionInput(
        ticket=run.ticket, triage=run.models_triage(), allowed_tools=list(ACTION_TOOLS)
    )
    facts_start = len(run.facts)
    deferred_start = len(run.deferred)
    tools = [run.business_tool(name, "action") for name in payload.allowed_tools]
    tools.append(_propose_tool(run))
    summary, flagged = await _subagent(
        run, SubagentName.ACTION, prompts.ACTION, payload, ActionSummary, tools
    )
    run.action_summary = summary.summary
    facts = run.facts[facts_start:] + [queued(a) for a in run.deferred[deferred_start:]]
    result = ActionResult(outcomes=list(run.outcomes), facts=facts, summary=summary.summary)
    return result, flagged


async def review(run: RequestRun, draft: AnswerDraft) -> EscalationDecision:
    """The escalation reviewer. Runs before every finalize; its screen is the request's last."""
    actions = ActionResult(
        outcomes=list(run.outcomes),
        facts=list(run.facts) + [queued(a) for a in run.deferred],
        summary=run.action_summary,
    )
    payload = EscalationInput(
        ticket=run.ticket, triage=run.models_triage(), answer=draft, actions=actions
    )
    decision, _ = await _subagent(
        run, SubagentName.ESCALATION, prompts.ESCALATION, payload, EscalationDecision, []
    )
    return decision


def queued(action: ProposedAction) -> str:
    return f"Proposed {action.tool.value} {canonical_json(action.arguments)}: awaiting policy check"


def _propose_tool(run: RequestRun) -> LocalTool:
    async def handler(arguments: JsonObject) -> tuple[str, bool]:
        try:
            proposal = ProposedAction.model_validate(arguments)
            if proposal.tool not in IRREVERSIBLE_TOOLS:
                return prompts.PROPOSE_ONLY_IRREVERSIBLE, True
            proposal_model(proposal.tool).model_validate(proposal.arguments)
        except ValidationError as exc:
            return validation_message(exc), True
        run.defer(proposal)
        return prompts.DEFERRED, False

    definition = ToolDefinition(
        name="propose_action",
        description=prompts.PROPOSE_DESCRIPTION,
        input_schema=_PROPOSE_SCHEMA,
    )
    return LocalTool(definition=definition, handler=handler)


def _propose_schema() -> JsonObject:
    """``ProposedAction`` with ``arguments`` narrowed to what each proposable tool takes.

    The model's own schema types ``arguments`` as recursive JSON, which Gemini rejects
    with HTTP 400 ("ref loops are only supported if they include optional ..."). The
    handler still validates against ``ProposedAction`` and ``proposal_model``.
    """
    tools = sorted(t.value for t in IRREVERSIBLE_TOOLS)
    shapes = [proposal_model(ToolName(t)).model_json_schema() for t in tools]
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "tool": {"type": "string", "enum": tools},
            "arguments": shapes[0] if len(shapes) == 1 else {"anyOf": shapes},
            "rationale": ProposedAction.model_json_schema()["properties"]["rationale"],
        },
        "required": ["tool", "arguments", "rationale"],
    }
    return cast(JsonObject, schema)


_PROPOSE_SCHEMA = _propose_schema()
