"""Code-written customer text: outcome notes, handoff summaries, the final response, and
redaction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from triagewright.agent import subagents
from triagewright.agent.outcomes import _needs_info, _refused
from triagewright.agent.runtime import RequestRun
from triagewright.contracts.agents import (
    ActionOutcome,
    ActionStatus,
    EscalationDecision,
    FinalResponse,
    LoopStop,
    ResolutionStatus,
)
from triagewright.contracts.common import JsonValue
from triagewright.contracts.ports import Guardrails, RequestTelemetry
from triagewright.contracts.rag import Citation
from triagewright.contracts.telemetry import SpanKind
from triagewright.contracts.tickets import Ticket
from triagewright.contracts.tools import ErrorCategory, RefundReason

_SUMMARY_MAX = 2000


class Replies:
    """The coordinator's methods that write customer and handoff text."""

    _guardrails: Guardrails
    _mode: Literal["live", "replay", "stub"]

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
            handoff_id=run.handoff.escalation_id if run.handoff else None,
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
            # Before the facts, so a long facts list can't cut the reasons off.
            f"Why escalated: {'; '.join(why) or 'reviewer decision'}",
            f"Attempted: {'; '.join(attempted) or 'no actions'}",
            f"Facts checked: {facts}",
        ]
        return "\n".join(lines)[:_SUMMARY_MAX]

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
                    "metadata": self._redact_metadata(ticket.metadata),
                }
            )

    def _redact_metadata(self, metadata: Mapping[str, str]) -> dict[str, str]:
        """Redact keys and values. Keys that redact alike stay distinct, so all get screened."""
        redacted: dict[str, str] = {}
        suffix: dict[str, int] = {}  # the last suffix tried per key, so collisions stay linear
        for key, value in metadata.items():
            base = name = self._guardrails.redact(key).text
            while name in redacted:
                suffix[base] = suffix.get(base, 1) + 1
                name = f"{base}#{suffix[base]}"
            redacted[name] = self._guardrails.redact(value).text
        return redacted


def _money(cents: JsonValue) -> str:
    if isinstance(cents, int) and not isinstance(cents, bool):
        return f"${cents / 100:,.2f}"
    return "the requested amount"
