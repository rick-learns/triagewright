"""Stub ticket resolver that escalates every ticket."""

from __future__ import annotations

from triagewright.contracts.agents import (
    EscalationDecision,
    FinalResponse,
    ResolutionStatus,
)
from triagewright.contracts.common import stable_id
from triagewright.contracts.tickets import Ticket
from triagewright.contracts.world import (
    EscalationQueue,
    EscalationReason,
    Priority,
)


class StubResolver:
    """Escalates every ticket. Lets API/CLI/evals run before the agent lands."""

    async def resolve(self, ticket: Ticket) -> FinalResponse:
        return FinalResponse(
            request_id=stable_id("req", ticket.ticket_id),
            ticket_id=ticket.ticket_id,
            status=ResolutionStatus.ESCALATED,
            reply="Thanks for reaching out. A support specialist will follow up shortly.",
            escalation=EscalationDecision(
                escalate=True,
                reasons=[EscalationReason.OUT_OF_SCOPE],
                queue=EscalationQueue.TIER1,
                priority=Priority.NORMAL,
                handoff_summary="Stub resolver: no agent wired yet.",
            ),
            mode="stub",
        )
