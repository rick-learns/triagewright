"""Escalation and routing: anonymous tickets, reviewer escalations, and handoff summaries."""

from __future__ import annotations

import pytest

from tests.agent.helpers import (
    FakeClock,
    RedactingGuardrails,
    TimedModel,
    call,
    finalize,
    refund_script,
    refund_ticket,
    review,
    stack,
    text,
    tool_results,
    triage,
)
from triagewright.agent import prompts
from triagewright.contracts.agents import (
    LoopConfig,
    LoopStop,
    ResolutionStatus,
)
from triagewright.contracts.evals import ToolFault
from triagewright.contracts.llm import StopReason
from triagewright.contracts.tools import ErrorCategory, ToolError, ToolName
from triagewright.testing import (
    sample_ticket,
)

pytestmark = pytest.mark.anyio


# ---- anonymous tickets -------------------------------------------------------------


async def test_anonymous_ticket_can_still_be_handed_off() -> None:
    s = stack(
        [
            triage("other", needs_human=True, reasons=["customer_requested_human"]),
            finalize("Sure, a person will help."),
            review(True, "tier1", ["customer_requested_human"], summary="Wants a human."),
        ]
    )
    response = await s.resolve(sample_ticket("Let me talk to a person.", customer_id=None))
    assert response.status is ResolutionStatus.ESCALATED
    (record,) = s.calls(ToolName.CREATE_ESCALATION)
    assert not record.result.is_error
    assert [e.ticket_id for e in s.gateway.ledger().escalations] == ["tkt_0001"]
    assert "support team" in response.reply


async def test_anonymous_ticket_needing_order_data_asks_for_info() -> None:
    s = stack(
        [
            triage(),
            call("get_order", {"order_id": "ord_a1002"}),
            finalize("Please sign in so I can look up your order.", "needs_info"),
            review(),
        ]
    )
    response = await s.resolve(sample_ticket(customer_id=None))
    (record,) = s.calls(ToolName.GET_ORDER)
    assert record.result.error is not None
    assert record.result.error.category is ErrorCategory.PERMISSION
    assert response.status is ResolutionStatus.NEEDS_INFO
    first = s.requests("coordinator")[0].messages[0].content[0]
    assert first.type == "text" and '"customer_in_scope":false' in first.text


async def test_anonymous_refund_proposal_asks_for_info() -> None:
    s = stack(refund_script("ord_a1001", 5000))
    response = await s.resolve(sample_ticket("Refund ord_a1001 please.", customer_id=None))
    assert response.status is ResolutionStatus.NEEDS_INFO
    assert "signed in" in response.reply


# ---- escalation rows driven by the reviewer ----------------------------------------


@pytest.mark.parametrize(
    ("body", "queue", "reasons", "priority"),
    [
        ("My charger is swelling and smells burnt.", "tier2", ["legal_or_safety"], "urgent"),
        ("I was charged twice for ord_a1001.", "billing", ["policy_limit"], "normal"),
        ("Please remove two-factor from my account.", "security", ["account_security"], "high"),
        ("Delete all my personal data.", "tier2", ["legal_or_safety"], "normal"),
    ],
)
async def test_reviewer_escalations_reach_the_gateway(
    body: str, queue: str, reasons: list[str], priority: str
) -> None:
    s = stack(
        [
            triage("other", needs_human=True),
            finalize("Thanks for letting us know."),
            review(True, queue, reasons, priority, "Goal; facts; attempted; why."),
        ]
    )
    response = await s.resolve(sample_ticket(body))
    assert response.status is ResolutionStatus.ESCALATED
    args = s.escalation_args()
    assert (args["queue"], args["reason_codes"], args["priority"]) == (queue, reasons, priority)
    assert args["ticket_id"] == "tkt_0001"
    assert response.escalation is not None and response.escalation.escalate


async def test_failed_escalation_write_does_not_promise_a_follow_up() -> None:
    error = ToolError(category=ErrorCategory.INTERNAL, message="db down", retryable=False)
    fault = ToolFault(tool=ToolName.CREATE_ESCALATION, error=error)
    s = stack(
        [triage(), finalize("Thanks."), review(True, "tier1", ["customer_requested_human"])],
        faults=[fault],
    )
    response = await s.resolve(sample_ticket("A human, please."))
    assert response.status is ResolutionStatus.ESCALATED
    assert response.reply.endswith(prompts.LOGGED_FOR_REVIEW)
    assert "passed your request" not in response.reply


async def test_a_full_reviewer_summary_still_fits_the_code_reasons() -> None:
    error = ToolError(category=ErrorCategory.TRANSIENT, message="busy", retryable=True)
    s = stack(
        [
            triage(),
            call("get_order", {"order_id": "ord_a1002"}),
            finalize(),
            review(True, "tier1", ["customer_requested_human"], summary="x" * 2000),
        ],
        faults=[ToolFault(tool=ToolName.GET_ORDER, error=error, times=5)],
        config=LoopConfig(retry_backoff_s=0),
    )
    response = await s.resolve(sample_ticket())
    summary = str(s.escalation_args()["summary"])
    assert len(summary) == 2000
    assert summary.endswith("Also: tool failed (get_order).")
    assert response.escalation is not None and response.escalation.handoff_summary == summary


async def test_redaction_cannot_push_the_code_reasons_out_of_the_summary() -> None:
    error = ToolError(category=ErrorCategory.TRANSIENT, message="busy", retryable=True)
    guard = RedactingGuardrails({"a@b.co": "[EMAIL_ADDRESS_1]"})
    s = stack(
        [
            triage(),
            call("get_order", {"order_id": "ord_a1002"}),
            finalize(),
            review(True, "tier1", ["customer_requested_human"], summary="a@b.co " * 285),
        ],
        guardrails=guard,
        faults=[ToolFault(tool=ToolName.GET_ORDER, error=error, times=5)],
        config=LoopConfig(retry_backoff_s=0),
    )
    await s.resolve(sample_ticket())
    summary = str(s.escalation_args()["summary"])
    assert len(summary) <= 2000
    assert summary.endswith("Also: tool failed (get_order).")


async def test_retryable_permission_errors_are_tool_failures() -> None:
    error = ToolError(category=ErrorCategory.PERMISSION, message="token expired", retryable=True)
    fault = ToolFault(tool=ToolName.CHECK_REFUND_ELIGIBILITY, error=error, times=5)
    s = stack(
        refund_script("ord_a1001", 5000), faults=[fault], config=LoopConfig(retry_backoff_s=0)
    )
    response = await s.resolve(refund_ticket("cus_ava01", "ord_a1001"))
    assert response.status is ResolutionStatus.ESCALATED
    assert s.escalation_args()["reason_codes"] == ["tool_failure"]
    assert "confirm the order" not in response.reply


@pytest.mark.parametrize(
    ("category", "retryable", "raw"),
    [(ErrorCategory.TRANSIENT, True, 3), (ErrorCategory.INTERNAL, False, 1)],
)
async def test_a_failed_tool_stays_failed_for_the_request(
    category: ErrorCategory, retryable: bool, raw: int
) -> None:
    """A model asking again gets the same failure without another raw call."""
    error = ToolError(category=category, message="down", retryable=retryable)
    lookup = call("get_order", {"order_id": "ord_a1002"})
    s = stack(
        [triage(), lookup, lookup, finalize(), review()],
        faults=[ToolFault(tool=ToolName.GET_ORDER, error=error, times=10)],
        config=LoopConfig(retry_backoff_s=0),
    )
    response = await s.resolve(sample_ticket())
    assert len(s.calls(ToolName.GET_ORDER)) == raw
    first, again = (tool_results(r)[-1] for r in s.requests("coordinator")[1:3])
    assert again.content == first.content and again.is_error
    assert response.status is ResolutionStatus.ESCALATED
    args = s.escalation_args()
    assert (args["queue"], args["reason_codes"]) == ("tier1", ["tool_failure"])


async def test_a_bad_call_is_not_a_tool_failure() -> None:
    error = ToolError(category=ErrorCategory.VALIDATION, message="bad id", retryable=False)
    lookup = call("get_order", {"order_id": "ord_a1002"})
    s = stack(
        [triage(), lookup, lookup, finalize(), review()],
        faults=[ToolFault(tool=ToolName.GET_ORDER, error=error)],
    )
    response = await s.resolve(sample_ticket())
    assert len(s.calls(ToolName.GET_ORDER)) == 2
    assert response.status is ResolutionStatus.RESOLVED


async def test_reviewer_escalation_without_queue_or_reasons_gets_defaults() -> None:
    s = stack([triage(), finalize(), review(True)])
    await s.resolve(sample_ticket())
    args = s.escalation_args()
    assert (args["queue"], args["reason_codes"]) == ("tier1", ["low_confidence"])
    assert str(args["summary"]).startswith("Customer goal:")


# ---- inactive accounts -----------------------------------------------------------


async def test_a_locked_sign_in_on_a_suspended_account_goes_to_tier2() -> None:
    """Only get_customer shows the suspension, so code reads it when no model did."""
    s = stack(
        [
            triage("account_access", intent="Get back into a locked account"),
            call("get_account", {"customer_id": "cus_eli05"}),
            finalize("Use the password reset link to unlock your account."),
            review(escalate=True, queue="security", reasons=["account_security"]),
        ]
    )
    ticket = sample_ticket("My account is locked. How do I get back in?", customer_id="cus_eli05")
    response = await s.resolve(ticket)
    assert [c.caller for c in s.calls(ToolName.GET_CUSTOMER)] == ["coordinator"]
    assert response.status is ResolutionStatus.ESCALATED
    args = s.escalation_args()
    assert args["queue"] == "tier2"
    assert "account_inactive" in str(args["reason_codes"])
    assert "account is suspended" in str(args["summary"])
    # The self-serve draft is ruled out by the status, so code writes the reply.
    assert response.reply.startswith(prompts.FALLBACK_REPLY)
    assert "password reset" not in response.reply


async def test_a_lockout_on_an_active_account_stays_self_serve() -> None:
    s = stack(
        [
            triage("account_access", intent="Get back into a locked account"),
            call("get_customer", {"customer_id": "cus_ava01"}),
            finalize("Use the password reset link to unlock your account."),
            review(),
        ]
    )
    response = await s.resolve(sample_ticket("My account is locked. How do I get back in?"))
    assert len(s.calls(ToolName.GET_CUSTOMER)) == 1  # the model's read is reused
    assert response.status is ResolutionStatus.RESOLVED
    assert s.calls(ToolName.CREATE_ESCALATION) == []


async def test_a_fallback_keeps_a_known_suspended_account_in_tier2() -> None:
    s = stack(
        [
            triage("account_access", intent="Get back into a locked account"),
            call("get_customer", {"customer_id": "cus_eli05"}),
        ],
        config=LoopConfig(max_turns=1),
    )
    ticket = sample_ticket("My account is locked. How do I get back in?", customer_id="cus_eli05")
    response = await s.resolve(ticket)
    assert response.stop is LoopStop.MAX_TURNS
    assert s.escalation_args()["queue"] == "tier2"


@pytest.mark.parametrize(
    "category", [ErrorCategory.NOT_FOUND, ErrorCategory.INTERNAL], ids=lambda c: c.value
)
async def test_an_unreadable_account_status_is_a_tool_failure(category: ErrorCategory) -> None:
    error = ToolError(category=category, message="no answer", retryable=False)
    s = stack(
        [
            triage("account_access", intent="Get back into a locked account"),
            finalize("Use the password reset link to unlock your account."),
            review(),
        ],
        faults=[ToolFault(tool=ToolName.GET_CUSTOMER, error=error, times=5)],
    )
    response = await s.resolve(sample_ticket("My account is locked. How do I get back in?"))
    assert len(s.calls(ToolName.GET_CUSTOMER)) == 1
    assert response.status is ResolutionStatus.ESCALATED
    args = s.escalation_args()
    assert args["queue"] == "tier1"
    assert "tool_failure" in str(args["reason_codes"])
    assert response.reply.startswith(prompts.FALLBACK_REPLY)  # the draft can't be confirmed


async def test_a_provider_refusal_still_routes_a_suspended_account_to_tier2() -> None:
    s = stack(
        [
            triage("account_access", intent="Get back into a locked account"),
            text("I can't help with that.", StopReason.REFUSAL),
        ]
    )
    ticket = sample_ticket("My account is locked. How do I get back in?", customer_id="cus_eli05")
    response = await s.resolve(ticket)
    assert response.stop is LoopStop.REFUSAL
    assert response.status is ResolutionStatus.ESCALATED
    assert s.escalation_args()["queue"] == "tier2"
    assert response.reply.startswith(prompts.FALLBACK_REPLY)


async def test_a_provider_refusal_on_an_active_account_stays_refused() -> None:
    s = stack(
        [
            triage("account_access", intent="Get back into a locked account"),
            text("I can't help with that.", StopReason.REFUSAL),
        ]
    )
    response = await s.resolve(sample_ticket("My account is locked. How do I get back in?"))
    assert response.status is ResolutionStatus.REFUSED
    assert s.calls(ToolName.CREATE_ESCALATION) == []


async def test_a_status_read_past_the_budget_after_a_refusal_hands_off() -> None:
    clock = FakeClock()
    model = TimedModel(
        [
            triage("account_access", intent="Get back into a locked account"),
            text("I can't help with that.", StopReason.REFUSAL),
        ],
        clock=clock,
        step_s=14,
    )
    s = stack(model=model, clock=clock, config=LoopConfig(total_timeout_s=30, handoff_reserve_s=3))
    ticket = sample_ticket("My account is locked. How do I get back in?", customer_id="cus_eli05")
    response = await s.resolve(ticket)
    assert response.stop is LoopStop.TIMEOUT
    assert response.status is ResolutionStatus.ESCALATED
    assert s.calls(ToolName.GET_CUSTOMER) == []
    assert s.escalation_args()["queue"] == "tier1"
