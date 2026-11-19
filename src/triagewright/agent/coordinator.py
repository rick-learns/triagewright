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
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import Field

from triagewright.agent import prompts, subagents
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
    ActionOutcome,
    ActionStatus,
    AnswerDraft,
    EscalationDecision,
    FinalResponse,
    LoopConfig,
    LoopStop,
    ResolutionStatus,
    SubagentName,
    TicketCategory,
    TriageResult,
)
from triagewright.contracts.common import (
    ChunkId,
    Contract,
    JsonObject,
    JsonValue,
    canonical_json,
    stable_id,
)
from triagewright.contracts.guardrails import ContentSource
from triagewright.contracts.llm import ToolDefinition
from triagewright.contracts.policy import REFUND_OUTCOME
from triagewright.contracts.ports import (
    ActionExecutor,
    Guardrails,
    ModelClient,
    RequestTelemetry,
    TelemetryFactory,
    ToolGateway,
)
from triagewright.contracts.rag import Citation
from triagewright.contracts.telemetry import SpanKind
from triagewright.contracts.tickets import Ticket
from triagewright.contracts.tools import (
    TOOL_SPECS,
    CreateEscalationInput,
    ErrorCategory,
    RefundReason,
    SideEffect,
    ToolName,
)
from triagewright.contracts.world import EscalationQueue, EscalationReason, Priority

PURPOSES: tuple[str, ...] = ("coordinator", *(name.value for name in SubagentName))

# The coordinator may call every read-only business tool directly.
READ_TOOLS: tuple[ToolName, ...] = tuple(
    name for name, spec in TOOL_SPECS.items() if spec.side_effect is SideEffect.READ
)

# With an injection in the ticket, these intents get the security handoff (behavior.md).
_SENSITIVE = frozenset({TicketCategory.REFUND_REQUEST, TicketCategory.ACCOUNT_ACCESS})
# Errors the customer can fix by giving other details; anything else is a tool failure.
_ASK_CUSTOMER = frozenset(
    {ErrorCategory.NOT_FOUND, ErrorCategory.VALIDATION, ErrorCategory.PERMISSION}
)
_SUMMARY_MAX = 2000

_NO_ARGS: JsonObject = {"type": "object", "properties": {}, "additionalProperties": False}


class CitationCandidate(Contract):
    chunk_id: ChunkId
    quote: Annotated[str, Field(min_length=1, max_length=400)]


class FinalizeArgs(Contract):
    """What the coordinator model may supply. Everything else in the response is code's."""

    status: Literal["resolved", "needs_info", "refused"]
    reply: Annotated[str, Field(min_length=1, max_length=4000)]
    citations: list[CitationCandidate] = []


class CoordinatorInput(Contract):
    ticket: Ticket
    triage: TriageResult
    customer_in_scope: bool


class Coordinator:
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
        texts = [run.ticket.subject, run.ticket.body, *run.ticket.metadata.values()]
        run.ticket_flagged = run.screen("\n".join(texts), ContentSource.CUSTOMER)
        try:
            run.triage = await subagents.triage(run)
            run.ticket_flagged = run.ticket_flagged or run.triage.injection_suspected
            run.injection = run.injection or run.ticket_flagged
            final = await self._coordinate(run)
            citations = run.cite_all([(c.chunk_id, c.quote) for c in final.citations])
            try:
                review = await subagents.review(run, self._draft(run, final, citations))
            except Refused:
                raise Fallback(
                    LoopStop.ERROR, EscalationReason.LOW_CONFIDENCE, "escalation reviewer refused"
                ) from None
            await self._propose_deferred(run)
        except Refused:
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
        )

    def _tools(self, run: RequestRun) -> list[LocalTool]:
        async def answer(_: JsonObject) -> tuple[str, bool]:
            draft, flagged = await subagents.answer(run)
            return as_data(canonical_json(draft.model_dump(mode="json")), flagged), False

        async def action(_: JsonObject) -> tuple[str, bool]:
            result, flagged = await subagents.action(run)
            return as_data(canonical_json(result.model_dump(mode="json")), flagged), False

        tools = [run.business_tool(name, "coordinator") for name in READ_TOOLS]
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
        """The finalize draft as the reviewer sees it."""
        assert run.triage is not None
        sub = run.answer
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
            failed = outcome.status is ActionStatus.FAILED and outcome.result is not None
            if failed and outcome.result is not None and not _needs_info(outcome):
                run.tool_failures.append(outcome.result.tool)
            run.ensure_time()

    # ---- routing and the response ------------------------------------------------

    def _route(
        self, run: RequestRun, review: EscalationDecision | None
    ) -> EscalationDecision | None:
        """Code's escalation decision. The reviewer can add an escalation, never remove one."""
        queue: EscalationQueue | None = None
        priority = Priority.NORMAL
        reasons: list[EscalationReason] = []
        why: list[str] = []
        for outcome in run.outcomes:
            winning = _refund_reason(outcome)
            routed = REFUND_OUTCOME[winning] if winning else None
            if routed is not None and queue is None:
                queue = routed
                inactive = winning is RefundReason.ACCOUNT_INACTIVE
                reasons.append(
                    EscalationReason.ACCOUNT_INACTIVE if inactive else EscalationReason.POLICY_LIMIT
                )
                why.append(f"refund blocked by policy ({winning})")
        reviewed = review is not None and review.escalate
        if review is not None and reviewed:
            queue = queue or review.queue or EscalationQueue.TIER1
            priority = review.priority
            reasons.extend(review.reasons)
        if run.tool_failures:
            queue = queue or EscalationQueue.TIER1
            reasons.append(EscalationReason.TOOL_FAILURE)
            failed = sorted({t.value for t in run.tool_failures})
            why.append(f"tool kept failing after retries ({', '.join(failed)})")
        category = run.triage.category if run.triage else None
        forced_write = run.ticket_flagged and category in _SENSITIVE
        if run.injection and (queue is not None or run.irreversible_requested or forced_write):
            queue = EscalationQueue.SECURITY
            reasons.insert(0, EscalationReason.INJECTION_SUSPECTED)
            why.insert(0, "possible prompt injection alongside a write or handoff request")
        if queue is None:
            return None
        summary = review.handoff_summary if review is not None and reviewed else ""
        if summary and why:
            also = f"\nAlso: {'; '.join(why)}."[:_SUMMARY_MAX]
            summary = summary[: _SUMMARY_MAX - len(also)] + also
        return EscalationDecision(
            escalate=True,
            reasons=list(dict.fromkeys(reasons)) or [EscalationReason.LOW_CONFIDENCE],
            queue=queue,
            priority=priority,
            handoff_summary=self._redact(summary or self._summary(run, why))[:_SUMMARY_MAX],
        )

    async def _respond(
        self,
        run: RequestRun,
        reply: str,
        proposed: ResolutionStatus,
        citations: list[Citation],
        review: EscalationDecision | None,
        stop: LoopStop,
    ) -> FinalResponse:
        decision = self._route(run, review)
        notes = [note for o in run.outcomes if (note := self._note(run, o))]
        if decision is not None:
            status = ResolutionStatus.ESCALATED
            notes.append(await self._handoff(run, decision))
        elif any(o.status is ActionStatus.PENDING_APPROVAL for o in run.outcomes):
            status = ResolutionStatus.PENDING_APPROVAL
        elif any(_refused(o) for o in run.outcomes):
            status = ResolutionStatus.REFUSED
        elif any(_needs_info(o) for o in run.outcomes):
            status = ResolutionStatus.NEEDS_INFO
        else:
            status = proposed
        return self._final(run, status, reply, notes, citations, decision, stop)

    async def _fallback(self, run: RequestRun, fallback: Fallback) -> FinalResponse:
        """The safe fallback: hand off to a human, never a half-finished action."""
        queue = EscalationQueue.TIER1
        reasons = [fallback.reason]
        why = [fallback.detail]
        if run.tool_failures:
            reasons.append(EscalationReason.TOOL_FAILURE)
        if run.injection:
            queue = EscalationQueue.SECURITY
            reasons.insert(0, EscalationReason.INJECTION_SUSPECTED)
            why.insert(0, "possible prompt injection")
        decision = EscalationDecision(
            escalate=True,
            reasons=list(dict.fromkeys(reasons)),
            queue=queue,
            priority=Priority.NORMAL,
            handoff_summary=self._redact(self._summary(run, why))[:_SUMMARY_MAX],
        )
        notes = [note for o in run.outcomes if (note := self._note(run, o))]
        notes.append(await self._handoff(run, decision))
        return self._final(
            run,
            ResolutionStatus.ESCALATED,
            prompts.FALLBACK_REPLY,
            notes,
            [],
            decision,
            fallback.stop,
        )

    async def _handoff(self, run: RequestRun, decision: EscalationDecision) -> str:
        """Code-owned ``create_escalation`` for the trusted ticket id. Returns the reply note."""
        assert decision.queue is not None
        args = CreateEscalationInput(
            ticket_id=run.ticket.ticket_id,
            queue=decision.queue,
            priority=decision.priority,
            summary=decision.handoff_summary or "Escalated by the support agent.",
            reason_codes=decision.reasons,
        )
        try:
            result = await run.call_tool(
                ToolName.CREATE_ESCALATION,
                args.model_dump(mode="json"),
                "coordinator",
                handoff=True,
            )
        except Fallback:
            return prompts.LOGGED_FOR_REVIEW
        return prompts.LOGGED_FOR_REVIEW if result.is_error else prompts.handed_off(decision.queue)

    def _final(
        self,
        run: RequestRun,
        status: ResolutionStatus,
        reply: str,
        notes: list[str],
        citations: list[Citation],
        decision: EscalationDecision | None,
        stop: LoopStop,
    ) -> FinalResponse:
        return FinalResponse(
            request_id=run.request_id,
            ticket_id=run.ticket.ticket_id,
            status=status,
            reply=self._redact("\n\n".join([reply, *notes]), run.telemetry),
            citations=citations,
            triage=run.models_triage() if run.triage else None,
            actions=list(run.outcomes),
            escalation=decision,
            stop=stop,
            turns=run.turns.get("coordinator", 0),
            mode=self._mode,
        )

    def _note(self, run: RequestRun, outcome: ActionOutcome) -> str | None:
        """The customer-facing outcome of one proposal. Models never state outcomes."""
        args = outcome.action.arguments
        order = args.get("order_id")
        if outcome.status is ActionStatus.PENDING_APPROVAL:
            return (
                f"Your refund request of {_money(args.get('amount_cents'))} for order {order} "
                "has been submitted for review. A team member reviews every refund before "
                "it's issued."
            )
        if _refused(outcome):
            elig = outcome.eligibility
            assert elig is not None
            match elig.reasons[0]:
                case RefundReason.NOT_DELIVERED:
                    return f"Order {order} can't be refunded because it hasn't been delivered."
                case RefundReason.OUTSIDE_WINDOW if elig.window_ends_at is not None:
                    ended = elig.window_ends_at.date().isoformat()
                    return f"Order {order} can't be refunded: its refund window ended on {ended}."
                case RefundReason.EXCEEDS_BALANCE:
                    left = _money(elig.max_refundable_cents)
                    return f"That's more than the refundable balance on order {order} ({left})."
            return f"Order {order} can't be refunded under our refund policy."
        if _needs_info(outcome):
            assert outcome.result is not None and outcome.result.error is not None
            if run.ticket.customer_id is None:
                return (
                    "I can't look up orders without knowing your account. Please write to us "
                    "while signed in to the account that placed the order."
                )
            if outcome.result.error.category is ErrorCategory.NOT_FOUND:
                return (
                    f"I couldn't find order {order} on your account. Could you check the order id?"
                )
            return "I couldn't process that refund request. Could you confirm the order and amount?"
        return None

    def _summary(self, run: RequestRun, why: list[str]) -> str:
        """A handoff summary a human can act on without rereading the ticket."""
        goal = run.triage.intent if run.triage else (run.ticket.subject or run.ticket.body)
        facts = "; ".join(f[:300] for f in run.facts) or "none"
        attempted = [f"{o.action.tool.value} -> {o.status.value}" for o in run.outcomes]
        attempted += [f"{subagents.queued(a)} (not submitted)" for a in run.deferred]
        lines = [
            f"Customer goal: {goal[:300]}",
            f"Facts checked: {facts}",
            f"Attempted: {'; '.join(attempted) or 'no actions'}",
            f"Why escalated: {'; '.join(why) or 'reviewer decision'}",
        ]
        return "\n".join(lines)[:_SUMMARY_MAX]

    # ---- redaction ---------------------------------------------------------------

    def _redact(self, text: str, telemetry: RequestTelemetry | None = None) -> str:
        if telemetry is None:
            return self._guardrails.redact(text).text
        with telemetry.span("guardrails.redact", SpanKind.GUARDRAIL):
            return self._guardrails.redact(text).text

    def _redact_ticket(self, ticket: Ticket, telemetry: RequestTelemetry) -> Ticket:
        """Customer text is redacted before any model sees it."""
        with telemetry.span("guardrails.redact", SpanKind.GUARDRAIL):
            return ticket.model_copy(
                update={
                    "body": self._guardrails.redact(ticket.body).text,
                    "subject": self._guardrails.redact(ticket.subject).text,
                    "metadata": {
                        k: self._guardrails.redact(v).text for k, v in ticket.metadata.items()
                    },
                }
            )


def _refund_reason(outcome: ActionOutcome) -> RefundReason | None:
    """The winning reason of a policy-blocked refund (reasons are in precedence order)."""
    elig = outcome.eligibility
    if outcome.status is ActionStatus.BLOCKED and elig is not None and elig.reasons:
        return elig.reasons[0]
    return None


def _refused(outcome: ActionOutcome) -> bool:
    winning = _refund_reason(outcome)
    return winning is not None and REFUND_OUTCOME[winning] is None


def _needs_info(outcome: ActionOutcome) -> bool:
    """A proposal that failed on its arguments, scope, or a missing record, not on the backend."""
    error = outcome.result.error if outcome.result else None
    if error is None or error.category not in _ASK_CUSTOMER:
        return False
    return outcome.status is ActionStatus.FAILED or (
        outcome.status is ActionStatus.BLOCKED and outcome.verdict is None
    )


def _money(cents: JsonValue) -> str:
    if isinstance(cents, int) and not isinstance(cents, bool):
        return f"${cents / 100:,.2f}"
    return "the requested amount"
