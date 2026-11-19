"""Routing and status: code's escalation decision, the response status, and the handoff."""

from __future__ import annotations

from triagewright.agent import prompts
from triagewright.agent.outcomes import _needs_info, _refund_reason, _refused
from triagewright.agent.refunds import RefundRules, _refund_proposed, _refund_requested
from triagewright.agent.replies import _SUMMARY_MAX, Replies
from triagewright.agent.runtime import Fallback, RequestRun
from triagewright.contracts.agents import (
    ActionStatus,
    EscalationDecision,
    FinalResponse,
    LoopStop,
    ResolutionStatus,
    TicketCategory,
)
from triagewright.contracts.policy import REFUND_OUTCOME
from triagewright.contracts.rag import Citation
from triagewright.contracts.tools import (
    CreateEscalationInput,
    GetCustomerInput,
    RefundReason,
    ToolName,
)
from triagewright.contracts.world import (
    CustomerStatus,
    EscalationQueue,
    EscalationReason,
    Priority,
)


class Routing(Replies, RefundRules):
    """The coordinator's routing methods."""

    def _route(
        self, run: RequestRun, review: EscalationDecision | None, proposed: ResolutionStatus
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
        if _inactive_account(run):
            # The suspended or closed row wins over self-serve lockout and the security row.
            queue = queue or EscalationQueue.TIER2
            reasons.append(EscalationReason.ACCOUNT_INACTIVE)
            why.append(f"account is {run.customer_status}")
        reviewed = (
            review is not None
            and review.escalate
            and not _left_to_code(run, review, refusing=proposed is ResolutionStatus.REFUSED)
        )
        if review is not None and reviewed:
            queue = queue or review.queue or EscalationQueue.TIER1
            priority = review.priority
            reasons.extend(review.reasons)
        if run.tool_failures:
            queue = queue or EscalationQueue.TIER1
            reasons.append(EscalationReason.TOOL_FAILURE)
            failed = sorted({t.value for t in run.tool_failures})
            why.append(f"tool kept failing after retries ({', '.join(failed)})")
        # A refund ask with injection suspected anywhere, the reviewer's screen included
        # (it runs last), goes to security whatever the models did. So does an injection
        # planted in a tool result or KB page: it targets the system, not the customer's ask.
        forced_write = run.triage is not None and (
            _refund_requested(run)
            or (
                run.ticket_flagged
                and EscalationReason.ACCOUNT_SECURITY in run.triage.needs_human_reasons
            )
        )
        other = queue is not None or run.irreversible_requested or forced_write
        if run.injection and (other or run.planted_in):
            queue = EscalationQueue.SECURITY
            reasons.insert(0, EscalationReason.INJECTION_SUSPECTED)
            alongside = " alongside a write or handoff request" if other else ""
            why.insert(0, _injection_why(run, alongside))
        if queue is None:
            return None
        given = review.handoff_summary if review is not None and reviewed else ""
        summary = self._redact(given or self._summary(run, why))
        if given and why:
            # Redacted before the cut, so redaction can't push code's reasons out.
            also = "\n" + self._redact(f"Also: {'; '.join(why)}.")[: _SUMMARY_MAX - 1]
            summary = summary[: _SUMMARY_MAX - len(also)] + also
        return EscalationDecision(
            escalate=True,
            reasons=list(dict.fromkeys(reasons)) or [EscalationReason.LOW_CONFIDENCE],
            queue=queue,
            priority=priority,
            handoff_summary=summary[:_SUMMARY_MAX],
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
        decision = self._route(run, review, proposed)
        notes = [note for o in run.outcomes if (note := self._note(run, o))]
        if run.triage is not None and (_refund_requested(run) or _refund_proposed(run)):
            reply = self._refund_reply(reply, decision)
        elif decision is not None and (
            run.planted_in or _inactive_account(run) or ToolName.GET_CUSTOMER in run.tool_failures
        ):
            # The models read the planted text, or drafted a self-serve answer the account's
            # status rules out or that its status can't confirm, so none of their words go out.
            reply = prompts.FALLBACK_REPLY
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
        if _inactive_account(run):
            queue = EscalationQueue.TIER2
            reasons.append(EscalationReason.ACCOUNT_INACTIVE)
            why.append(f"account is {run.customer_status}")
        if run.injection:
            queue = EscalationQueue.SECURITY
            reasons.insert(0, EscalationReason.INJECTION_SUSPECTED)
            why.insert(0, _injection_why(run))
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


def _inactive_account(run: RequestRun) -> bool:
    account = run.triage is not None and run.triage.category is TicketCategory.ACCOUNT_ACCESS
    inactive = (CustomerStatus.SUSPENDED, CustomerStatus.CLOSED)
    return account and run.customer_status in inactive


async def _check_account(run: RequestRun) -> None:
    """Read the customer's status for an account-access ticket if no model did: only
    ``get_customer`` shows a suspension, and it decides the queue."""
    customer = run.ticket.customer_id
    account = run.triage is not None and run.triage.category is TicketCategory.ACCOUNT_ACCESS
    if not account or customer is None:
        return
    if run.customer_status is None:
        args = GetCustomerInput(customer_id=customer).model_dump(mode="json")
        result = await run.call_tool(ToolName.GET_CUSTOMER, args, "coordinator")
        run.observe(result, args)
        if result.is_error and ToolName.GET_CUSTOMER not in run.tool_failures:
            # The status decides the queue, so not knowing it is a failure a human handles.
            run.tool_failures.append(ToolName.GET_CUSTOMER)


def _injection_why(run: RequestRun, alongside: str = "") -> str:
    where = f"; planted in {', '.join(dict.fromkeys(run.planted_in))}" if run.planted_in else ""
    return f"possible prompt injection{alongside}{where}"


def _left_to_code(run: RequestRun, review: EscalationDecision, *, refusing: bool) -> bool:
    """A review whose every reason is one code already handled adds nothing, unless triage
    found a human need code didn't handle:

    - ``approval_required`` once the request's one action is decided: a pending approval
      is that human step, and a blocked action has nothing to approve.
    - ``injection_suspected`` when code suspected it before the reviewer did and the draft
      refuses: the injection rule in ``_route`` still sends a write or handoff request to
      security.
    """
    handled: set[EscalationReason] = set()
    decided = (ActionStatus.PENDING_APPROVAL, ActionStatus.BLOCKED)
    if len(run.outcomes) == 1 and run.outcomes[0].status in decided:
        handled.add(EscalationReason.APPROVAL_REQUIRED)
    if run.flagged_before_review and refusing:
        handled.add(EscalationReason.INJECTION_SUSPECTED)
    reasons = set(review.reasons)
    if not reasons or not reasons <= handled:
        return False
    triage = run.triage
    other_need = triage is not None and (
        bool(set(triage.needs_human_reasons) - handled)
        or (triage.needs_human and not triage.needs_human_reasons)
    )
    return not other_need
