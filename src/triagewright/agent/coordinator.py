"""The coordinator: the ``TicketResolver`` behind the API, CLI, and evals (ADR-001 §3).

Flow for one ticket:

1. Redact the ticket and ``screen`` it. Run the triage subagent.
2. The coordinator loop: its tools are the answer and action subagents plus the
   read-only business tools, and it ends with a forced ``finalize`` call (reply,
   citation candidates, and a proposed status).
3. The escalation reviewer runs before every finalize. Its ``screen`` is the
   request's last, so ``injection_suspected`` is final after it. Code runs triage
   and the reviewer itself, so the model can't skip either.
4. Only then are deferred irreversible proposals sent to ``ActionExecutor.propose``.
5. Code routes the result: a blocked refund on
   ``policy.REFUND_OUTCOME[eligibility.reasons[0]]``, the reviewer's decision,
   persistent tool failures, and, first of all, the security handoff when
   injection is suspected. Code calls ``create_escalation`` (``caller="coordinator"``)
   and builds :class:`FinalResponse`; no model writes it.

Any bound (turns, per-turn or total time), a second truncation, or a second
invalid structured output gives the safe fallback: no pending proposal is sent,
and the ticket is escalated with a code-built handoff summary.

Modules: ``finalize`` (the finalize tool's arguments), ``refunds`` (refund rules),
``outcomes`` (reading executor outcomes), ``replies`` (code-written text and
redaction), and ``routing`` (escalation and status). :class:`Coordinator` runs the flow.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from typing import Literal

from triagewright.agent import prompts, subagents
from triagewright.agent.finalize import CitationCandidate as CitationCandidate
from triagewright.agent.finalize import FinalizeArgs as FinalizeArgs
from triagewright.agent.outcomes import _ASK_CUSTOMER as _ASK_CUSTOMER
from triagewright.agent.outcomes import _needs_info as _needs_info
from triagewright.agent.outcomes import _refund_reason as _refund_reason
from triagewright.agent.outcomes import _refused as _refused
from triagewright.agent.refunds import _ORDER_ID as _ORDER_ID
from triagewright.agent.refunds import _POLICY_SEARCH as _POLICY_SEARCH
from triagewright.agent.refunds import _REFUND_POLICY_DOC as _REFUND_POLICY_DOC
from triagewright.agent.refunds import _order_unclear as _order_unclear
from triagewright.agent.refunds import _policy_citation as _policy_citation
from triagewright.agent.refunds import _refund_decidable as _refund_decidable
from triagewright.agent.refunds import _refund_proposed as _refund_proposed
from triagewright.agent.refunds import _refund_requested as _refund_requested
from triagewright.agent.refunds import _settled as _settled
from triagewright.agent.refunds import first_sentence as first_sentence
from triagewright.agent.replies import _SUMMARY_MAX as _SUMMARY_MAX
from triagewright.agent.replies import _money as _money
from triagewright.agent.routing import Routing, _check_account
from triagewright.agent.routing import _left_to_code as _left_to_code
from triagewright.agent.runtime import (
    Clock,
    Fallback,
    LocalTool,
    Refused,
    RequestRun,
    Sleep,
    as_data,
)
from triagewright.contracts.agents import (
    ActionStatus,
    AnswerDraft,
    FinalResponse,
    LoopConfig,
    LoopStop,
    ResolutionStatus,
    SubagentName,
    TriageResult,
)
from triagewright.contracts.common import Contract, JsonObject, canonical_json, stable_id
from triagewright.contracts.guardrails import ContentSource
from triagewright.contracts.llm import ToolDefinition
from triagewright.contracts.ports import (
    ActionExecutor,
    Guardrails,
    ModelClient,
    TelemetryFactory,
    ToolGateway,
)
from triagewright.contracts.rag import Citation
from triagewright.contracts.telemetry import SpanKind
from triagewright.contracts.tickets import Ticket
from triagewright.contracts.tools import TOOL_SPECS, SideEffect, ToolName
from triagewright.contracts.world import EscalationReason

PURPOSES: tuple[str, ...] = ("coordinator", *(name.value for name in SubagentName))

# The coordinator may call every read-only business tool directly.
READ_TOOLS: tuple[ToolName, ...] = tuple(
    name for name, spec in TOOL_SPECS.items() if spec.side_effect is SideEffect.READ
)
# On a refund request it gets only the knowledge base: the order lookups and the refund
# proposal belong to the action subagent, which it must run (enforced in ``_check_final``).
REFUND_READ_TOOLS: tuple[ToolName, ...] = (ToolName.SEARCH_KB,)

_NO_ARGS: JsonObject = {"type": "object", "properties": {}, "additionalProperties": False}


class CoordinatorInput(Contract):
    ticket: Ticket
    triage: TriageResult
    customer_in_scope: bool


class Coordinator(Routing):
    """Implements ``ports.TicketResolver`` over the model, tool, guardrail, and executor ports.

    ``models`` maps each purpose (``coordinator`` and every :class:`SubagentName`)
    to a model id. ``clock`` and ``sleep`` drive the time bounds and retry backoff.
    """

    def __init__(
        self,
        *,
        model: ModelClient,
        gateway: ToolGateway,
        guardrails: Guardrails,
        executor: ActionExecutor,
        telemetry: TelemetryFactory,
        models: Mapping[str, str],
        config: LoopConfig | None = None,
        mode: Literal["live", "replay", "stub"] = "replay",
        clock: Clock = time.monotonic,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        missing = [p for p in PURPOSES if p not in models]
        if missing:
            raise ValueError(f"no model id for {missing}")
        self._model = model
        self._gateway = gateway
        self._guardrails = guardrails
        self._executor = executor
        self._telemetry = telemetry
        self._models = dict(models)
        self._config = config or LoopConfig()
        self._mode: Literal["live", "replay", "stub"] = mode
        self._clock = clock
        self._sleep = sleep

    async def resolve(self, ticket: Ticket) -> FinalResponse:
        request_id = stable_id("req", ticket.ticket_id)
        telemetry = self._telemetry.for_request(request_id, ticket.ticket_id)
        with telemetry.span("resolve", SpanKind.REQUEST, {"ticket_id": ticket.ticket_id}) as span:
            run = RequestRun(
                ticket=ticket,
                request_id=request_id,
                model=self._model,
                gateway=self._gateway,
                guardrails=self._guardrails,
                executor=self._executor,
                telemetry=telemetry,
                models=self._models,
                config=self._config,
                clock=self._clock,
                sleep=self._sleep,
                started=self._clock(),
            )
            run.ticket = self._redact_ticket(ticket, telemetry)
            response = await self._resolve(run)
            span.set_attribute("status", response.status.value)
            span.set_attribute("stop", response.stop.value)
        return response.model_copy(update={"metrics": telemetry.finish()})

    # ---- flow --------------------------------------------------------------------

    async def _resolve(self, run: RequestRun) -> FinalResponse:
        # Metadata reaches the models with the ticket, so it is screened with it.
        meta = [f"{k}: {v}" for k, v in run.ticket.metadata.items()]
        texts = [run.ticket.subject, run.ticket.body, *meta]
        run.ticket_flagged = run.screen("\n".join(texts), ContentSource.CUSTOMER)
        try:
            run.triage = await subagents.triage(run)
            run.ticket_flagged = run.ticket_flagged or run.triage.injection_suspected
            run.injection = run.injection or run.ticket_flagged
            final = self._own_refund_reply(run, await self._coordinate(run))
            await _check_account(run)  # before the review, so the reviewer sees the status
            citations = run.cite_all([(c.chunk_id, c.quote) for c in final.citations])
            run.flagged_before_review = run.injection
            try:
                review = await subagents.review(run, self._draft(run, final, citations))
            except Refused:
                raise Fallback(
                    LoopStop.ERROR, EscalationReason.LOW_CONFIDENCE, "escalation reviewer refused"
                ) from None
            if review.escalate and EscalationReason.INJECTION_SUSPECTED in review.reasons:
                run.injection = True  # the reviewer's suspicion blocks writes like a screen's
            await self._propose_deferred(run)
        except Refused:
            # A provider refusal isn't a scope decision: the account's status still routes.
            try:
                await _check_account(run)
            except Fallback as fallback:
                return await self._fallback(run, fallback)
            return await self._respond(
                run, prompts.REFUSAL_REPLY, ResolutionStatus.REFUSED, [], None, LoopStop.REFUSAL
            )
        except Fallback as fallback:
            return await self._fallback(run, fallback)
        proposed = ResolutionStatus(final.status)
        return await self._respond(
            run, final.reply, proposed, citations, review, LoopStop.COMPLETED
        )

    async def _coordinate(self, run: RequestRun) -> FinalizeArgs:
        assert run.triage is not None
        payload = CoordinatorInput(
            ticket=run.ticket,
            triage=run.models_triage(),
            customer_in_scope=run.ticket.customer_id is not None,
        )
        return await run.run_loop(
            "coordinator",
            prompts.COORDINATOR,
            run.data_prompt(payload),
            self._tools(run),
            FinalizeArgs,
            "finalize",
            prompts.FINALIZE_DESCRIPTION,
            check=lambda final: self._check_final(run, final),
            settle=lambda: _settled(run),
        )

    def _tools(self, run: RequestRun) -> list[LocalTool]:
        async def answer(_: JsonObject) -> tuple[str, bool]:
            draft, flagged = await subagents.answer(run)
            return as_data(canonical_json(draft.model_dump(mode="json")), flagged), False

        async def action(_: JsonObject) -> tuple[str, bool]:
            try:
                result, flagged = await subagents.action(run)
            except Refused:  # an internal step, not the customer's answer: hand off
                raise Fallback(
                    LoopStop.ERROR, EscalationReason.LOW_CONFIDENCE, "action subagent refused"
                ) from None
            return as_data(canonical_json(result.model_dump(mode="json")), flagged), False

        reads = REFUND_READ_TOOLS if _refund_requested(run) else READ_TOOLS
        tools = [run.business_tool(name, "coordinator") for name in reads]
        for name, description, handler in (
            (SubagentName.ANSWER, prompts.ANSWER_TOOL_DESCRIPTION, answer),
            (SubagentName.ACTION, prompts.ACTION_TOOL_DESCRIPTION, action),
        ):
            definition = ToolDefinition(
                name=name.value, description=description, input_schema=_NO_ARGS
            )
            tools.append(LocalTool(definition=definition, handler=handler))
        return tools

    def _draft(
        self, run: RequestRun, final: FinalizeArgs, citations: list[Citation]
    ) -> AnswerDraft:
        """The finalize draft as the reviewer sees it. Beside a refund proposal the reply is
        code's (``_own_refund_reply``), so an earlier answer's grounding doesn't describe it."""
        assert run.triage is not None
        sub = None if _refund_proposed(run) else run.answer
        return AnswerDraft(
            answer=final.reply,
            citations=citations,
            grounded=sub.grounded if sub else bool(citations),
            unanswerable=sub.unanswerable if sub else False,
            confidence=sub.confidence if sub else run.triage.confidence,
        )

    async def _propose_deferred(self, run: RequestRun) -> None:
        """Send queued irreversible proposals, now that ``injection_suspected`` is final.

        The budget is checked around each proposal: one in flight is never cancelled, its
        outcome is kept, and an overrun hands the ticket off.
        """
        run.ensure_time()  # the reviewer may have used up the budget
        while run.deferred:
            run.ensure_time()
            proposal = run.deferred.pop(0)
            with run.telemetry.span(
                "executor.propose", SpanKind.APPROVAL, {"tool": proposal.tool.value}
            ) as span:
                outcome = await self._executor.propose(
                    proposal, run.context("action"), injection_suspected=run.injection
                )
                span.set_attribute("status", outcome.status.value)
            run.outcomes.append(outcome)
            failed = outcome.status is ActionStatus.FAILED
            if failed and outcome.result is not None and not _needs_info(outcome):
                run.tool_failures.append(outcome.result.tool)
            run.ensure_time()
