"""Refund outcomes: approval, policy blocks and refusals, executor lookups, and the
reviewer's approval escalations."""

from __future__ import annotations

from typing import Any

import pytest

from tests.agent.helpers import (
    POLICY_CITE,
    SUBMITTED,
    FakeClock,
    call,
    finalize,
    propose_refund,
    refund_script,
    refund_ticket,
    review,
    stack,
    text,
    together,
    tool_results,
    triage,
)
from triagewright.agent import prompts
from triagewright.contracts.agents import (
    ActionOutcome,
    ActionStatus,
    LoopConfig,
    LoopStop,
    ProposedAction,
    ResolutionStatus,
)
from triagewright.contracts.evals import ToolFault
from triagewright.contracts.guardrails import (
    GuardrailDecision,
    GuardrailVerdict,
    PolicyContext,
)
from triagewright.contracts.tools import ErrorCategory, ToolError, ToolName
from triagewright.testing import (
    PassthroughGuardrails,
    sample_ticket,
    seed_world,
)

pytestmark = pytest.mark.anyio


# ---- refunds -----------------------------------------------------------------------


async def test_eligible_refund_goes_to_pending_approval() -> None:
    s = stack(refund_script("ord_a1001", 5000))
    response = await s.resolve(refund_ticket("cus_ava01", "ord_a1001"))
    assert response.status is ResolutionStatus.PENDING_APPROVAL
    assert response.escalation is None
    (outcome,) = response.actions
    assert outcome.status is ActionStatus.PENDING_APPROVAL
    (approval,) = s.approvals.list_pending()
    assert approval.approval_id == outcome.approval_id
    assert approval.arguments["amount_cents"] == 5000
    assert s.calls(ToolName.ISSUE_REFUND) == []
    assert len(s.gateway.ledger().refunds) == len(seed_world().refunds)
    (check,) = s.calls(ToolName.CHECK_REFUND_ELIGIBILITY)
    assert check.caller == "action"
    assert "$50.00 for order ord_a1001 has been submitted for review" in response.reply
    assert [c.doc_id for c in response.citations] == ["kb_refund_policy"]
    # The proposal and the cited policy settle the finalize: no model writes it, and the
    # reply is code's.
    assert (len(s.requests("action")), len(s.requests("coordinator"))) == (2, 2)
    assert response.reply.startswith(prompts.REFUND_OUTCOME_REPLY)


async def test_escalated_wins_over_pending_approval() -> None:
    s = stack(
        refund_script(
            "ord_a1001",
            5000,
            escalate=True,
            queue="tier1",
            reasons=["customer_requested_human"],
            summary="Wants a person. Refund of $50 for ord_a1001 is pending approval.",
        )
    )
    response = await s.resolve(refund_ticket("cus_ava01", "ord_a1001"))
    assert response.status is ResolutionStatus.ESCALATED
    assert [o.status for o in response.actions] == [ActionStatus.PENDING_APPROVAL]
    assert len(s.approvals.list_pending()) == 1
    args = s.escalation_args()
    assert args["summary"] == "Wants a person. Refund of $50 for ord_a1001 is pending approval."


async def test_an_approval_only_escalation_leaves_a_pending_refund_alone() -> None:
    # The approval queue is the human step; the reviewer asking for one adds nothing.
    script = refund_script("ord_a1001", 5000)
    s = stack([*script[:-1], review(True, "tier2", ["approval_required"])])
    response = await s.resolve(refund_ticket("cus_ava01", "ord_a1001"))
    assert response.status is ResolutionStatus.PENDING_APPROVAL
    assert response.escalation is None
    assert s.calls(ToolName.CREATE_ESCALATION) == []


async def test_an_approval_escalation_with_another_reason_still_escalates() -> None:
    script = refund_script("ord_a1001", 5000)
    reasons = ["approval_required", "customer_requested_human"]
    s = stack([*script[:-1], review(True, "tier1", reasons)])
    response = await s.resolve(refund_ticket("cus_ava01", "ord_a1001"))
    assert response.status is ResolutionStatus.ESCALATED


async def test_an_approval_escalation_keeps_a_human_need_triage_found() -> None:
    _, *rest = refund_script("ord_a1001", 5000)
    wants_person = triage(
        "refund_request",
        intent="Get a refund and talk to a person",
        needs_human=True,
        reasons=["customer_requested_human"],
    )
    s = stack([wants_person, *rest[:-1], review(True, "tier1", ["approval_required"])])
    response = await s.resolve(refund_ticket("cus_ava01", "ord_a1001"))
    assert response.status is ResolutionStatus.ESCALATED


async def test_an_approval_escalation_keeps_an_unexplained_human_need() -> None:
    _, *rest = refund_script("ord_a1001", 5000)
    unexplained = triage("refund_request", intent="Refund, and a person", needs_human=True)
    s = stack([unexplained, *rest[:-1], review(True, "tier1", ["approval_required"])])
    response = await s.resolve(refund_ticket("cus_ava01", "ord_a1001"))
    assert response.status is ResolutionStatus.ESCALATED


async def test_refunds_proposed_over_several_action_turns_are_all_kept() -> None:
    # A queued proposal doesn't end the action run: the next turn may propose another.
    triaged, search, act, propose, submitted, reviewed = refund_script("ord_h8001", 11800)
    second = propose_refund("ord_h8001", 2000)
    s = stack([triaged, search, act, propose, second, submitted, reviewed])
    response = await s.resolve(refund_ticket("cus_hal08", "ord_h8001"))
    amounts = {str(a.arguments["amount_cents"]) for a in s.approvals.list_pending()}
    assert amounts == {"2000", "11800"}
    assert response.status is ResolutionStatus.PENDING_APPROVAL


async def test_an_approval_escalation_with_two_actions_still_escalates() -> None:
    triaged, search, act, propose, submitted, _ = refund_script("ord_a1001", 5000)
    both = together(propose, propose_refund("ord_a1001", 2000))
    s = stack([triaged, search, act, both, submitted, review(True, "tier1", ["approval_required"])])
    response = await s.resolve(refund_ticket("cus_ava01", "ord_a1001"))
    assert [o.status for o in response.actions] == [ActionStatus.PENDING_APPROVAL] * 2
    assert response.status is ResolutionStatus.ESCALATED


async def test_an_approval_escalation_with_nothing_pending_still_escalates() -> None:
    s = stack(
        [
            triage("account_access", intent="Wants a person to approve a change"),
            finalize("We'll look into it.", "resolved"),
            review(True, "tier1", ["approval_required"]),
        ]
    )
    response = await s.resolve(sample_ticket("Please have someone approve my address change."))
    assert response.status is ResolutionStatus.ESCALATED


class MisleadingRuleIds(PassthroughGuardrails):
    """Blocks ineligible refunds with a rule id that names the wrong reason."""

    def check_action(self, action: ProposedAction, ctx: PolicyContext) -> GuardrailVerdict:
        verdict = super().check_action(action, ctx)
        if verdict.decision is GuardrailDecision.BLOCK:
            return verdict.model_copy(update={"rule_ids": ["refund.outside_window"]})
        return verdict


async def test_closed_and_expired_refund_routes_to_tier2_on_the_first_reason() -> None:
    s = stack(refund_script("ord_i9001", 1200), guardrails=MisleadingRuleIds())
    response = await s.resolve(refund_ticket("cus_ivo09", "ord_i9001"))
    (outcome,) = response.actions
    assert outcome.status is ActionStatus.BLOCKED
    assert outcome.eligibility is not None
    assert [r.value for r in outcome.eligibility.reasons] == ["account_inactive", "outside_window"]
    assert response.status is ResolutionStatus.ESCALATED
    args = s.escalation_args()
    assert (args["queue"], args["reason_codes"]) == ("tier2", ["account_inactive"])
    assert "Customer goal: Get a refund" in str(args["summary"])
    assert "specialist support team" in response.reply
    assert [c.doc_id for c in response.citations] == ["kb_refund_policy"]


async def test_an_approval_escalation_leaves_a_policy_refusal_alone() -> None:
    # Nothing to approve: the policy refused the refund, and code owns that outcome.
    *script, _ = refund_script("ord_b2001", 11800)
    s = stack([*script, review(True, "tier2", ["approval_required"])])
    response = await s.resolve(refund_ticket("cus_ben02", "ord_b2001"))
    assert response.status is ResolutionStatus.REFUSED
    assert s.calls(ToolName.CREATE_ESCALATION) == []


ANONYMOUS_REFUND = sample_ticket("Refund ord_h8001 please.", customer_id=None)
ASK_SIGN_IN = finalize("Please sign in.", "needs_info")
ASK_SIGN_IN_CITED = finalize("Please sign in.", "needs_info", POLICY_CITE)


async def test_refund_reply_without_a_policy_citation_is_repaired() -> None:
    # A proposed refund settles with code's citation; a reply without one must cite.
    triaged, search, *_ = refund_script("ord_h8001", 11800)
    s = stack([triaged, ASK_SIGN_IN, search, ASK_SIGN_IN_CITED, review()])
    response = await s.resolve(ANONYMOUS_REFUND)
    assert response.status is ResolutionStatus.NEEDS_INFO
    assert [c.doc_id for c in response.citations] == ["kb_refund_policy"]
    repair = s.requests("coordinator")[1]
    assert tool_results(repair)[-1].content == prompts.CITE_REFUND_POLICY
    assert repair.tool_choice is None  # free to search the knowledge base


async def test_citation_repair_after_a_forced_finalize_can_search() -> None:
    triaged, search, *_ = refund_script("ord_h8001", 11800)
    s = stack([triaged, text("Done."), ASK_SIGN_IN, search, ASK_SIGN_IN_CITED, review()])
    response = await s.resolve(ANONYMOUS_REFUND)
    assert [c.doc_id for c in response.citations] == ["kb_refund_policy"]
    assert s.requests("coordinator")[1].tool_choice is not None  # forced
    assert s.requests("coordinator")[-2].tool_choice is None


async def test_refund_reply_never_citing_the_policy_falls_back() -> None:
    s = stack([triage("refund_request"), ASK_SIGN_IN, ASK_SIGN_IN])
    response = await s.resolve(ANONYMOUS_REFUND)
    assert response.status is ResolutionStatus.ESCALATED
    assert response.stop is LoopStop.ERROR
    assert s.approvals.list_pending() == []


async def test_outside_window_refund_is_refused() -> None:
    s = stack(refund_script("ord_b2001", 11800))
    response = await s.resolve(refund_ticket("cus_ben02", "ord_b2001"))
    assert response.status is ResolutionStatus.REFUSED
    assert response.escalation is None
    assert s.calls(ToolName.CREATE_ESCALATION) == []
    assert "refund window ended on 2026-06-01" in response.reply


@pytest.mark.parametrize(
    ("customer", "order", "amount", "expected"),
    [
        ("cus_ava01", "ord_a1002", 2400, "hasn't been delivered"),
        ("cus_ava01", "ord_a1001", 12900, "refundable balance on order ord_a1001 ($99.00)"),
    ],
)
async def test_policy_refusals_explain_the_reason(
    customer: str, order: str, amount: int, expected: str
) -> None:
    s = stack(refund_script(order, amount))
    response = await s.resolve(refund_ticket(customer, order))
    assert response.status is ResolutionStatus.REFUSED
    assert expected in response.reply


async def test_over_hard_limit_routes_to_billing() -> None:
    s = stack(refund_script("ord_c3001", 89900))
    response = await s.resolve(refund_ticket("cus_cho03", "ord_c3001"))
    assert response.status is ResolutionStatus.ESCALATED
    args = s.escalation_args()
    assert (args["queue"], args["reason_codes"]) == ("billing", ["policy_limit"])


async def test_reviewer_cannot_undo_a_code_escalation() -> None:
    s = stack(refund_script("ord_c3001", 89900, escalate=False))
    response = await s.resolve(refund_ticket("cus_cho03", "ord_c3001"))
    assert response.status is ResolutionStatus.ESCALATED


async def test_refund_for_another_customers_order_needs_info() -> None:
    s = stack(refund_script("ord_b2001", 1000))
    response = await s.resolve(refund_ticket("cus_ava01", "ord_b2001"))
    assert response.status is ResolutionStatus.NEEDS_INFO
    assert "couldn't find order ord_b2001" in response.reply
    assert s.approvals.list_pending() == []


async def test_persistent_failure_in_the_executor_lookup_escalates() -> None:
    error = ToolError(category=ErrorCategory.TIMEOUT, message="slow", retryable=True)
    fault = ToolFault(tool=ToolName.CHECK_REFUND_ELIGIBILITY, error=error, times=3)
    s = stack(refund_script("ord_a1001", 5000), faults=[fault])
    response = await s.resolve(refund_ticket("cus_ava01", "ord_a1001"))
    assert len(s.calls(ToolName.CHECK_REFUND_ELIGIBILITY)) == 3
    assert response.status is ResolutionStatus.ESCALATED
    args = s.escalation_args()
    assert (args["queue"], args["reason_codes"]) == ("tier1", ["tool_failure"])


async def test_an_approval_escalation_on_a_failed_action_reaches_the_human() -> None:
    error = ToolError(category=ErrorCategory.INTERNAL, message="bug", retryable=False)
    fault = ToolFault(tool=ToolName.CHECK_REFUND_ELIGIBILITY, error=error)
    *script, _ = refund_script("ord_a1001", 5000)
    asks = review(True, "tier1", ["approval_required"], summary="Needs a person to approve.")
    s = stack([*script, asks], faults=[fault])
    response = await s.resolve(refund_ticket("cus_ava01", "ord_a1001"))
    assert [o.status for o in response.actions] == [ActionStatus.FAILED]
    args = s.escalation_args()
    assert args["reason_codes"] == ["approval_required", "tool_failure"]


async def test_backend_error_in_the_executor_lookup_is_a_tool_failure() -> None:
    error = ToolError(category=ErrorCategory.INTERNAL, message="bug", retryable=False)
    fault = ToolFault(tool=ToolName.CHECK_REFUND_ELIGIBILITY, error=error)
    s = stack(refund_script("ord_a1001", 5000), faults=[fault])
    response = await s.resolve(refund_ticket("cus_ava01", "ord_a1001"))
    assert response.status is ResolutionStatus.ESCALATED
    args = s.escalation_args()
    assert (args["queue"], args["reason_codes"]) == ("tier1", ["tool_failure"])
    assert "couldn't find" not in response.reply


async def test_slow_proposals_stop_at_the_budget_and_keep_their_outcome() -> None:
    clock = FakeClock()
    s = stack(
        [
            triage("refund_request"),
            call("action"),
            together(propose_refund("ord_a1001", 5000), propose_refund("ord_a1001", 3000)),
            SUBMITTED,
            review(),
        ],
        clock=clock,
        config=LoopConfig(total_timeout_s=30),
    )
    executor = s.coordinator._executor
    propose = executor.propose

    async def slow(*args: Any, **kwargs: Any) -> ActionOutcome:
        clock.now += 40
        return await propose(*args, **kwargs)

    executor.propose = slow  # type: ignore[method-assign]
    response = await s.resolve(refund_ticket("cus_ava01", "ord_a1001"))
    assert response.stop is LoopStop.TIMEOUT
    assert response.status is ResolutionStatus.ESCALATED
    assert [o.status for o in response.actions] == [ActionStatus.PENDING_APPROVAL]
    assert len(s.approvals.list_pending()) == 1
    assert "has been submitted for review" in response.reply
    assert "issue_refund -> pending_approval" in str(s.escalation_args()["summary"])
    assert "(not submitted)" in str(s.escalation_args()["summary"])


async def test_propose_action_rejects_bad_proposals() -> None:
    s = stack(
        [
            triage("refund_request"),
            call("action"),
            call("propose_action", {"tool": "issue_refund", "arguments": {}, "rationale": "x"}),
            call("propose_action", {"tool": "get_order", "arguments": {}, "rationale": "x"}),
            call("propose_action", {"tool": "create_escalation", "arguments": {}, "rationale": ""}),
            together(propose_refund("ord_a1001", 5000), propose_refund("ord_a1001", 5000)),
            SUBMITTED,
            review(),
        ]
    )
    response = await s.resolve(refund_ticket("cus_ava01", "ord_a1001"))
    results = [r for req in s.requests("action")[1:4] for r in tool_results(req)[-1:]]
    assert [r.is_error for r in results] == [True, True, True]
    assert results[0].content.startswith("Invalid arguments.")
    assert results[1].content == prompts.PROPOSE_ONLY_IRREVERSIBLE
    assert len(response.actions) == 1  # the duplicate proposal is sent once
