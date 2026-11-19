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
)
from triagewright.contracts.policy import REFUND_OUTCOME
from triagewright.contracts.rag import Citation
from triagewright.contracts.tools import CreateEscalationInput, RefundReason, ToolName
from triagewright.contracts.world import EscalationQueue, EscalationReason, Priority


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
        # (it runs last), goes to security whatever the models did.
        forced_write = run.triage is not None and (
            _refund_requested(run)
            or (
                run.ticket_flagged
                and EscalationReason.ACCOUNT_SECURITY in run.triage.needs_human_reasons
            )
        )
        if run.injection and (queue is not None or run.irreversible_requested or forced_write):
            queue = EscalationQueue.SECURITY
            reasons.insert(0, EscalationReason.INJECTION_SUSPECTED)
            why.insert(0, "possible prompt injection alongside a write or handoff request")
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
            reply = self._refund_reply(run, reply, decision, notes)
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
