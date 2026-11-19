"""The generator's clock and refund policy, and the records it returns."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict

from triagewright.contracts.agents import ResolutionStatus, TicketCategory
from triagewright.contracts.guardrails import PolicyConfig
from triagewright.contracts.tickets import Ticket
from triagewright.contracts.tools import RefundReason
from triagewright.contracts.world import WorldState

# The seed world's clock (data/seed/meta.json), so generated records merge with it.
AS_OF = datetime(2026, 6, 15, 12, tzinfo=UTC)
# Generated refunds are built for the default policy ($500 cap, 30 days, 2 per 30 days).
POLICY = PolicyConfig()


class RefundOutcome(StrEnum):
    """What a generated refund request is built to hit: eligible, or one policy reason."""

    ELIGIBLE = "eligible"
    ACCOUNT_INACTIVE = RefundReason.ACCOUNT_INACTIVE.value
    NOT_DELIVERED = RefundReason.NOT_DELIVERED.value
    OUTSIDE_WINDOW = RefundReason.OUTSIDE_WINDOW.value
    EXCEEDS_BALANCE = RefundReason.EXCEEDS_BALANCE.value
    OVER_HARD_LIMIT = RefundReason.OVER_HARD_LIMIT.value
    REFUND_COUNT_LIMIT = RefundReason.REFUND_COUNT_LIMIT.value


class ExpectedRefund(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    order_id: str
    amount_cents: int
    outcome: RefundOutcome
    reasons: list[RefundReason]  # what evaluate_refund should return, in precedence order
    status: ResolutionStatus  # per docs/contracts/behavior.md "Refund decisions"
    queue: str | None


class Expected(BaseModel):
    """One ``expected.jsonl`` row: what a generated ticket was built to exercise."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ticket_id: str
    category: TicketCategory
    customer_id: str
    order_id: str | None
    refund: ExpectedRefund | None = None


@dataclass
class Generated:
    world: WorldState
    tickets: list[Ticket]
    expected: list[Expected]
    meta: dict[str, Any] = field(default_factory=dict)
