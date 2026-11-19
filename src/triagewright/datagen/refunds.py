"""Refund scenarios: an order and amount built to hit one policy outcome, and its expectation."""

from __future__ import annotations

from collections.abc import Sequence

from triagewright.contracts.agents import ResolutionStatus
from triagewright.contracts.policy import REFUND_OUTCOME
from triagewright.contracts.tools import RefundReason
from triagewright.contracts.world import Order, OrderStatus
from triagewright.datagen.builder import _Builder
from triagewright.datagen.models import POLICY, ExpectedRefund, RefundOutcome


def _refund_scenario(
    b: _Builder,
    outcome: RefundOutcome,
    normal: Sequence[str],
    inactive: Sequence[str],
    frequent: Sequence[str],
) -> tuple[str, Order, int]:
    """Build a fresh order (and pick its customer) so exactly ``outcome`` applies."""
    rng = b.rng
    window = POLICY.refund_window_days
    pool = {RefundOutcome.ACCOUNT_INACTIVE: inactive, RefundOutcome.REFUND_COUNT_LIMIT: frequent}
    cid = rng.choice(pool.get(outcome, normal))
    recent = rng.randint(1, window - 2)
    match outcome:
        case RefundOutcome.NOT_DELIVERED:
            order = b.order(cid, rng.choice([OrderStatus.PLACED, OrderStatus.SHIPPED]))
        case RefundOutcome.OUTSIDE_WINDOW:
            order = b.order(cid, OrderStatus.DELIVERED, delivered_days_ago=window + recent)
        case RefundOutcome.OVER_HARD_LIMIT:
            order = b.order(cid, OrderStatus.DELIVERED, delivered_days_ago=recent, premium=True)
        case _:
            order = b.order(cid, OrderStatus.DELIVERED, delivered_days_ago=recent)
    amount = order.total_cents
    if outcome is RefundOutcome.EXCEEDS_BALANCE:
        amount += rng.randint(5, 30) * 100
    elif outcome is RefundOutcome.ELIGIBLE and rng.random() < 0.3:
        amount //= 2  # the cheapest regular item is 900 cents, so this never drops below 100
    return cid, order, amount


def _expected_refund(order: Order, amount: int, outcome: RefundOutcome) -> ExpectedRefund:
    if outcome is RefundOutcome.ELIGIBLE:
        return ExpectedRefund(
            order_id=order.order_id,
            amount_cents=amount,
            outcome=outcome,
            reasons=[],
            status=ResolutionStatus.PENDING_APPROVAL,
            queue=None,
        )
    reason = RefundReason(outcome.value)
    queue = REFUND_OUTCOME[reason]
    return ExpectedRefund(
        order_id=order.order_id,
        amount_cents=amount,
        outcome=outcome,
        reasons=[reason],
        status=ResolutionStatus.REFUSED if queue is None else ResolutionStatus.ESCALATED,
        queue=None if queue is None else queue.value,
    )
