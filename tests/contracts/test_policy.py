from __future__ import annotations

from datetime import timedelta

from triagewright.contracts.common import canonical_json, stable_id
from triagewright.contracts.guardrails import PolicyConfig
from triagewright.contracts.policy import (
    REFUND_OUTCOME,
    REFUND_PRECEDENCE,
    approval_token,
    evaluate_refund,
    verify_approval_token,
)
from triagewright.contracts.tools import RefundReason
from triagewright.contracts.world import CustomerStatus, EscalationQueue, RefundStatus
from triagewright.testing import TEST_APPROVAL_KEY, seed_as_of, seed_world

WORLD = seed_world()
CUSTOMERS = {c.customer_id: c for c in WORLD.customers}
ORDERS = {o.order_id: o for o in WORLD.orders}


def evaluate(order_id: str, amount: int, **overrides: object) -> list[RefundReason]:
    order = ORDERS[order_id]
    customer = CUSTOMERS[order.customer_id]
    if "status" in overrides:
        customer = customer.model_copy(update={"status": overrides["status"]})
    return evaluate_refund(
        order=order,
        customer=customer,
        refunds=WORLD.refunds,
        amount_cents=amount,
        as_of=seed_as_of(),
        policy=PolicyConfig(),
    ).reasons


def test_every_reason_has_an_outcome_and_a_rank() -> None:
    assert set(REFUND_OUTCOME) == set(RefundReason) == set(REFUND_PRECEDENCE)
    assert REFUND_PRECEDENCE[0] is RefundReason.ACCOUNT_INACTIVE
    assert REFUND_OUTCOME[RefundReason.OUTSIDE_WINDOW] is None  # refuse
    assert REFUND_OUTCOME[RefundReason.OVER_HARD_LIMIT] is EscalationQueue.BILLING


def test_reasons_are_reported_in_precedence_order() -> None:
    assert evaluate("ord_h8001", 5000) == []
    assert evaluate("ord_i9001", 1200) == [
        RefundReason.ACCOUNT_INACTIVE,
        RefundReason.OUTSIDE_WINDOW,
    ]
    assert evaluate("ord_k1101", 1000) == [RefundReason.REFUND_COUNT_LIMIT]
    assert evaluate("ord_c3001", 89900) == [RefundReason.OVER_HARD_LIMIT]
    assert evaluate("ord_h8001", 5000, status=CustomerStatus.SUSPENDED) == [
        RefundReason.ACCOUNT_INACTIVE
    ]


def test_returned_orders_follow_the_standard_rule() -> None:
    # ord_h8003 is returned and already fully refunded, and its window has passed.
    assert evaluate("ord_h8003", 100) == [
        RefundReason.OUTSIDE_WINDOW,
        RefundReason.EXCEEDS_BALANCE,
    ]


def test_only_issued_refunds_in_30_days_count() -> None:
    order = ORDERS["ord_k1101"]
    customer = CUSTOMERS["cus_kai11"]
    old = [
        r.model_copy(update={"created_at": seed_as_of() - timedelta(days=31)})
        if r.customer_id == "cus_kai11"
        else r
        for r in WORLD.refunds
    ]
    reversed_ = [
        r.model_copy(update={"status": RefundStatus.REJECTED})
        if r.customer_id == "cus_kai11"
        else r
        for r in WORLD.refunds
    ]
    for refunds in (old, reversed_):
        elig = evaluate_refund(
            order=order,
            customer=customer,
            refunds=refunds,
            amount_cents=1000,
            as_of=seed_as_of(),
            policy=PolicyConfig(),
        )
        assert elig.eligible and elig.refunds_last_30d == 0


def test_approval_token_binds_every_field() -> None:
    fields = {
        "approval_id": "apr_0001",
        "order_id": "ord_h8001",
        "amount_cents": 5900,
        "idempotency_key": "refund-apr_0001",
    }
    token = approval_token(TEST_APPROVAL_KEY, **fields)  # type: ignore[arg-type]
    assert verify_approval_token(TEST_APPROVAL_KEY, token, **fields)  # type: ignore[arg-type]
    assert not verify_approval_token(b"other", token, **fields)  # type: ignore[arg-type]
    for key, value in [("amount_cents", 5901), ("order_id", "ord_h8002"), ("approval_id", "apr_2")]:
        assert not verify_approval_token(TEST_APPROVAL_KEY, token, **{**fields, key: value})  # type: ignore[arg-type]


def test_canonical_json_and_stable_ids_are_deterministic() -> None:
    assert canonical_json({"b": 1, "a": [1, {"d": 2, "c": 3}]}) == '{"a":[1,{"c":3,"d":2}],"b":1}'
    assert stable_id("apr", "tkt_0001", 1) == stable_id("apr", "tkt_0001", 1)
    assert stable_id("apr", "tkt_0001", 1) != stable_id("apr", "tkt_0001", 2)
    assert stable_id("esc", "x").startswith("esc_") and len(stable_id("esc", "x")) == 16
