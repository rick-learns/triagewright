from __future__ import annotations

from datetime import timedelta

import pytest

from triagewright.contracts.agents import ProposedAction
from triagewright.contracts.common import JsonObject
from triagewright.contracts.guardrails import (
    GuardrailDecision,
    GuardrailVerdict,
    PolicyConfig,
    PolicyContext,
)
from triagewright.contracts.policy import evaluate_refund
from triagewright.contracts.ports import Guardrails
from triagewright.contracts.tools import RefundEligibility, ToolName
from triagewright.guardrails import RuleGuardrails
from triagewright.testing import seed_as_of, seed_world

G = RuleGuardrails()
BLOCK = GuardrailDecision.BLOCK


def eligibility(
    order_id: str, amount: int, policy: PolicyConfig | None = None
) -> RefundEligibility:
    world = seed_world()
    order = next(o for o in world.orders if o.order_id == order_id)
    customer = next(c for c in world.customers if c.customer_id == order.customer_id)
    return evaluate_refund(
        order=order,
        customer=customer,
        refunds=world.refunds,
        amount_cents=amount,
        as_of=seed_as_of(),
        policy=PolicyConfig(),
    )


def refund(order_id: str = "ord_h8001", amount: int = 5900) -> ProposedAction:
    return ProposedAction(
        tool=ToolName.ISSUE_REFUND,
        arguments={"order_id": order_id, "amount_cents": amount, "reason": "damaged"},
        rationale="x",
    )


def pctx(
    elig: RefundEligibility | None = None,
    *,
    customer_id: str | None = "cus_hal08",
    injection: bool = False,
) -> PolicyContext:
    return PolicyContext(
        ticket_id="tkt_0001",
        customer_id=customer_id,
        eligibility=elig,
        injection_suspected=injection,
        now=seed_as_of(),
    )


def check(action: ProposedAction, ctx: PolicyContext) -> GuardrailVerdict:
    return G.check_action(action, ctx)


def test_satisfies_the_port() -> None:
    assert isinstance(G, Guardrails)


def test_eligible_refund_requires_approval() -> None:
    verdict = check(refund(), pctx(eligibility("ord_h8001", 5900)))
    assert verdict.decision is GuardrailDecision.REQUIRE_APPROVAL
    assert verdict.rule_ids == ["approval.irreversible"] and not verdict.injection_suspected


def test_code_owned_tools_are_blocked_first() -> None:
    escalation: JsonObject = {
        "ticket_id": "tkt_0001",
        "queue": "security",
        "priority": "high",
        "summary": "x",
        "reason_codes": ["security"],
    }
    action = ProposedAction(tool=ToolName.CREATE_ESCALATION, arguments=escalation, rationale="x")
    assert check(action, pctx(injection=True)).rule_ids == ["tool.code_owned"]


def test_invalid_or_smuggled_arguments_are_blocked() -> None:
    smuggled = ProposedAction(
        tool=ToolName.ISSUE_REFUND,
        arguments={**refund().arguments, "approval_token": "0" * 64},
        rationale="x",
    )
    verdict = check(smuggled, pctx(eligibility("ord_h8001", 5900)))
    assert verdict.decision is BLOCK and verdict.rule_ids == ["action.invalid_arguments"]
    bad_read = ProposedAction(tool=ToolName.GET_ORDER, arguments={}, rationale="x")
    assert check(bad_read, pctx()).rule_ids == ["action.invalid_arguments"]


def test_writes_are_blocked_once_injection_is_suspected() -> None:
    verdict = check(refund(), pctx(eligibility("ord_h8001", 5900), injection=True))
    assert verdict.decision is BLOCK and verdict.rule_ids == ["injection.write_blocked"]
    assert verdict.injection_suspected
    read = ProposedAction(
        tool=ToolName.GET_ORDER, arguments={"order_id": "ord_h8001"}, rationale=""
    )
    assert check(read, pctx(injection=True)).decision is GuardrailDecision.ALLOW


def test_irreversible_action_needs_a_customer() -> None:
    verdict = check(refund(), pctx(eligibility("ord_h8001", 5900), customer_id=None))
    assert verdict.rule_ids == ["scope.no_customer"]


def test_refund_without_eligibility_is_blocked() -> None:
    assert check(refund(), pctx()).rule_ids == ["refund.no_eligibility"]


@pytest.mark.parametrize(
    ("order_id", "amount", "rule_ids"),
    [
        ("ord_b2001", 1000, ["refund.outside_window"]),
        ("ord_a1002", 100, ["refund.not_delivered"]),
        ("ord_a1001", 12900, ["refund.exceeds_balance"]),
        ("ord_k1101", 1000, ["refund.refund_count_limit"]),
        ("ord_c3001", 89900, ["refund.over_hard_limit"]),
        ("ord_i9001", 1200, ["refund.account_inactive", "refund.outside_window"]),
        ("ord_e5001", 1000, ["refund.account_inactive"]),
    ],
)
def test_ineligible_refunds_are_blocked_in_policy_order(
    order_id: str, amount: int, rule_ids: list[str]
) -> None:
    elig = eligibility(order_id, amount)
    verdict = check(refund(order_id, amount), pctx(elig))
    assert verdict.decision is BLOCK and verdict.rule_ids == rule_ids
    assert verdict.reasons == [r.value for r in elig.reasons]


def test_eligibility_must_cover_the_proposed_refund() -> None:
    elig = eligibility("ord_h8001", 100)
    for action in (refund("ord_h8002", 100), refund("ord_h8001", 11801)):
        assert check(action, pctx(elig)).rule_ids == ["refund.eligibility_mismatch"]
    other_order = eligibility("ord_b2001", 1000)  # ineligible, but for another order
    verdict = check(refund("ord_h8001", 1000), pctx(other_order))
    assert verdict.rule_ids == ["refund.eligibility_mismatch"]


def test_policy_config_caps_the_amount() -> None:
    strict = RuleGuardrails(PolicyConfig(refund_hard_limit_cents=5000))
    elig = eligibility("ord_h8001", 5900)  # eligible under the default $500 limit
    verdict = strict.check_action(refund(amount=5900), pctx(elig))
    assert verdict.rule_ids == ["refund.eligibility_mismatch"]
    assert strict.check_action(refund(amount=5000), pctx(elig)).rule_ids == [
        "approval.irreversible"
    ]


def test_check_action_is_deterministic_and_ignores_the_clock() -> None:
    elig = eligibility("ord_h8001", 5900)
    later = pctx(elig).model_copy(update={"now": seed_as_of() + timedelta(days=365)})
    assert check(refund(), pctx(elig)) == check(refund(), later)


def test_reads_are_allowed() -> None:
    read = ProposedAction(
        tool=ToolName.LIST_ORDERS, arguments={"customer_id": "cus_hal08"}, rationale="x"
    )
    assert check(read, pctx()) == GuardrailVerdict(decision=GuardrailDecision.ALLOW)
