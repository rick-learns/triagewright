"""The one refund policy and the approval token. Pure functions shared by every lane.

The MCP server's ``check_refund_eligibility`` and ``issue_refund``, the
guardrails' ``check_action``, the fakes, and the eval cases all use
:func:`evaluate_refund`, so "eligible" means the same thing everywhere.
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Sequence
from datetime import datetime, timedelta

from triagewright.contracts.guardrails import PolicyConfig
from triagewright.contracts.tools import RefundEligibility, RefundReason
from triagewright.contracts.world import (
    Customer,
    CustomerStatus,
    EscalationQueue,
    Order,
    OrderStatus,
    Refund,
    RefundStatus,
)

REFUND_PRECEDENCE: tuple[RefundReason, ...] = tuple(RefundReason)

# What the agent does for the winning reason. None in the queue column means refuse.
REFUND_OUTCOME: dict[RefundReason, EscalationQueue | None] = {
    RefundReason.ACCOUNT_INACTIVE: EscalationQueue.TIER2,
    RefundReason.NOT_DELIVERED: None,
    RefundReason.OUTSIDE_WINDOW: None,
    RefundReason.EXCEEDS_BALANCE: None,
    RefundReason.OVER_HARD_LIMIT: EscalationQueue.BILLING,
    RefundReason.REFUND_COUNT_LIMIT: EscalationQueue.BILLING,
}

_REFUNDABLE = frozenset({OrderStatus.DELIVERED, OrderStatus.RETURNED})


def evaluate_refund(
    *,
    order: Order,
    customer: Customer,
    refunds: Sequence[Refund],
    amount_cents: int,
    as_of: datetime,
    policy: PolicyConfig,
) -> RefundEligibility:
    """Apply the refund policy. ``refunds`` may include other customers' rows; they're ignored."""
    issued = [
        r
        for r in refunds
        if r.customer_id == customer.customer_id and r.status is RefundStatus.ISSUED
    ]
    already = sum(r.amount_cents for r in issued if r.order_id == order.order_id)
    recent = sum(1 for r in issued if as_of - r.created_at <= timedelta(days=30))
    max_refundable = max(order.total_cents - already, 0)
    window_end = (
        order.delivered_at + timedelta(days=policy.refund_window_days)
        if order.delivered_at
        else None
    )
    failed: set[RefundReason] = set()
    if customer.status is not CustomerStatus.ACTIVE:
        failed.add(RefundReason.ACCOUNT_INACTIVE)
    if order.status not in _REFUNDABLE or order.delivered_at is None:
        failed.add(RefundReason.NOT_DELIVERED)
    if window_end is not None and as_of > window_end:
        failed.add(RefundReason.OUTSIDE_WINDOW)
    if amount_cents > max_refundable:
        failed.add(RefundReason.EXCEEDS_BALANCE)
    if amount_cents > policy.refund_hard_limit_cents:
        failed.add(RefundReason.OVER_HARD_LIMIT)
    if recent >= policy.max_refunds_per_customer_30d:
        failed.add(RefundReason.REFUND_COUNT_LIMIT)
    return RefundEligibility(
        order_id=order.order_id,
        eligible=not failed,
        max_refundable_cents=max_refundable,
        already_refunded_cents=already,
        refunds_last_30d=recent,
        window_ends_at=window_end,
        reasons=[r for r in REFUND_PRECEDENCE if r in failed],
    )


def approval_token(
    key: bytes, *, approval_id: str, order_id: str, amount_cents: int, idempotency_key: str
) -> str:
    """HMAC over exactly what a human approved. The MCP server refuses refunds without it."""
    msg = f"{approval_id}|{order_id}|{amount_cents}|{idempotency_key}".encode()
    return hmac.new(key, msg, hashlib.sha256).hexdigest()


def verify_approval_token(
    key: bytes,
    token: str,
    *,
    approval_id: str,
    order_id: str,
    amount_cents: int,
    idempotency_key: str,
) -> bool:
    expected = approval_token(
        key,
        approval_id=approval_id,
        order_id=order_id,
        amount_cents=amount_cents,
        idempotency_key=idempotency_key,
    )
    return hmac.compare_digest(expected, token)


def refund_idempotency_key(approval_id: str) -> str:
    """One approval can issue at most one refund: the key is derived from the approval."""
    return f"refund-{approval_id}"
