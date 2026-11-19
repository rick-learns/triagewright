"""The fake action executor: approvals, retries, blocking, and escalations."""

from __future__ import annotations

import pytest

from tests.contracts.fakes_helpers import CTX, ctx, executor, gateway
from triagewright.contracts.agents import ActionStatus, ProposedAction
from triagewright.contracts.common import JsonObject
from triagewright.contracts.evals import ToolFault
from triagewright.contracts.guardrails import ApprovalDecision, ApprovalStatus
from triagewright.contracts.policy import REFUND_OUTCOME, approval_token, refund_idempotency_key
from triagewright.contracts.tools import ErrorCategory, RefundReason, ToolError, ToolName
from triagewright.contracts.world import EscalationQueue
from triagewright.testing import (
    TEST_APPROVAL_KEY,
    FakeActionExecutor,
    InMemoryApprovalStore,
    PassthroughGuardrails,
    seed_as_of,
    seed_world,
)

pytestmark = pytest.mark.anyio


async def test_refund_needs_a_human_approval_and_runs_once() -> None:
    gw = gateway()
    ex = FakeActionExecutor(gw, PassthroughGuardrails(), InMemoryApprovalStore(), now=seed_as_of())
    hal = ctx("cus_hal08")
    proposal = ProposedAction(
        tool=ToolName.ISSUE_REFUND,
        arguments={"order_id": "ord_h8001", "amount_cents": 5900, "reason": "damaged"},
        rationale="damaged on arrival",
    )
    pending = await ex.propose(proposal, hal, injection_suspected=False)
    assert pending.status is ActionStatus.PENDING_APPROVAL and pending.approval_id
    assert gw.refunds == seed_world().refunds  # nothing issued in-request
    again = await ex.propose(proposal, hal, injection_suspected=False)
    assert again.approval_id == pending.approval_id  # deterministic id

    # A forged approval is refused by the gateway itself.
    forged: JsonObject = {
        **proposal.arguments,
        "idempotency_key": refund_idempotency_key(pending.approval_id),
        "approval_id": pending.approval_id,
        "approval_token": "0" * 64,
    }
    denied = await gw.call(ToolName.ISSUE_REFUND, forged, ctx("cus_hal08", "human"))
    assert denied.error and denied.error.category is ErrorCategory.PERMISSION
    not_yet = await ex.execute_approved(pending.approval_id)
    assert not_yet.status is ActionStatus.BLOCKED

    decision = ApprovalDecision(
        approval_id=pending.approval_id,
        approved=True,
        reviewer="supervisor",
        decided_at=seed_as_of(),
    )
    done = await ex.decide(decision)
    assert done.status is ActionStatus.EXECUTED
    assert len(gw.refunds) == len(seed_world().refunds) + 1
    stored = ex.approvals.get(pending.approval_id)
    assert stored and stored.status is ApprovalStatus.EXECUTED
    assert (await ex.execute_approved(pending.approval_id)).status is ActionStatus.BLOCKED


async def test_refund_retry_reuses_the_idempotency_key() -> None:
    fault = ToolFault(
        tool=ToolName.ISSUE_REFUND,
        error=ToolError(category=ErrorCategory.TIMEOUT, message="slow", retryable=True),
    )
    gw = gateway(faults=[fault])
    ex = executor(gw)
    proposal = ProposedAction(
        tool=ToolName.ISSUE_REFUND,
        arguments={"order_id": "ord_h8001", "amount_cents": 5900, "reason": "damaged"},
        rationale="x",
    )
    pending = await ex.propose(proposal, ctx("cus_hal08"), injection_suspected=False)
    assert pending.approval_id
    failed = await ex.decide(
        ApprovalDecision(
            approval_id=pending.approval_id, approved=True, reviewer="s", decided_at=seed_as_of()
        )
    )
    assert failed.status is ActionStatus.FAILED
    stored = ex.approvals.get(pending.approval_id)
    assert stored and stored.status is ApprovalStatus.APPROVED
    retried = await ex.execute_approved(pending.approval_id)
    assert retried.status is ActionStatus.EXECUTED
    assert stored.idempotency_key == refund_idempotency_key(pending.approval_id)


async def test_executor_blocks_unsafe_proposals() -> None:
    ex = executor()
    over_limit = ProposedAction(
        tool=ToolName.ISSUE_REFUND,
        arguments={"order_id": "ord_c3001", "amount_cents": 89900, "reason": "x"},
        rationale="x",
    )
    blocked = await ex.propose(over_limit, ctx("cus_cho03"), injection_suspected=False)
    assert blocked.verdict and blocked.verdict.rule_ids == ["refund.over_hard_limit"]
    ok = ProposedAction(
        tool=ToolName.ISSUE_REFUND,
        arguments={"order_id": "ord_h8001", "amount_cents": 100, "reason": "x"},
        rationale="x",
    )
    injected = await ex.propose(ok, ctx("cus_hal08"), injection_suspected=True)
    assert injected.status is ActionStatus.BLOCKED
    smuggled = ProposedAction(
        tool=ToolName.ISSUE_REFUND,
        arguments={**ok.arguments, "approval_id": "apr_0001"},
        rationale="x",
    )
    rejected = await ex.propose(smuggled, ctx("cus_hal08"), injection_suspected=False)
    assert rejected.status is ActionStatus.BLOCKED and rejected.result
    assert ex.approvals.list_pending() == []


ESCALATION: JsonObject = {
    "ticket_id": "tkt_0001",
    "queue": "billing",
    "priority": "high",
    "summary": "Refund over limit",
    "reason_codes": ["policy_limit"],
}


async def test_escalation_is_code_owned_and_works_without_a_customer() -> None:
    ex = executor()
    proposed = ProposedAction(tool=ToolName.CREATE_ESCALATION, arguments=ESCALATION, rationale="x")
    blocked = await ex.propose(proposed, CTX, injection_suspected=True)
    assert blocked.status is ActionStatus.BLOCKED and ex.gateway.escalations == []
    by_model = await ex.gateway.call(ToolName.CREATE_ESCALATION, ESCALATION, CTX)
    assert by_model.error and by_model.error.category is ErrorCategory.PERMISSION
    anon = ctx(None, "coordinator")
    first = await ex.gateway.call(ToolName.CREATE_ESCALATION, ESCALATION, anon)
    again = await ex.gateway.call(
        ToolName.CREATE_ESCALATION, {**ESCALATION, "queue": "security"}, anon
    )
    assert first.data and again.data == first.data  # idempotent per ticket
    assert len(ex.gateway.escalations) == 1
    other = await ex.gateway.call(
        ToolName.CREATE_ESCALATION, {**ESCALATION, "ticket_id": "tkt_9999"}, anon
    )
    assert other.error and other.error.category is ErrorCategory.PERMISSION


def _refund(order_id: str = "ord_h8001", amount: int = 5900) -> ProposedAction:
    return ProposedAction(
        tool=ToolName.ISSUE_REFUND,
        arguments={"order_id": order_id, "amount_cents": amount, "reason": "damaged"},
        rationale="x",
    )


def _transient(times: int) -> ToolFault:
    return ToolFault(
        tool=ToolName.CHECK_REFUND_ELIGIBILITY,
        error=ToolError(category=ErrorCategory.TRANSIENT, message="db", retryable=True),
        times=times,
    )


async def test_eligibility_lookup_retries_then_reports_the_error() -> None:
    gw = gateway(faults=[_transient(2)])
    ok = await executor(gw).propose(_refund(), ctx("cus_hal08"), injection_suspected=False)
    assert ok.status is ActionStatus.PENDING_APPROVAL
    assert [c.tool for c in gw.calls].count(ToolName.CHECK_REFUND_ELIGIBILITY) == 3

    gw = gateway(faults=[_transient(3)])
    failed = await executor(gw).propose(_refund(), ctx("cus_hal08"), injection_suspected=False)
    assert failed.status is ActionStatus.FAILED and failed.attempts == 3
    assert failed.result and failed.result.error and failed.result.error.retryable

    missing = await executor().propose(_refund("ord_zzzz"), CTX, injection_suspected=False)
    assert missing.status is ActionStatus.FAILED and missing.result and missing.result.error
    assert missing.result.error.category is ErrorCategory.NOT_FOUND


async def test_blocked_refund_carries_the_winning_reason() -> None:
    # Closed account and expired window: account_inactive wins, so tier2.
    out = await executor().propose(
        _refund("ord_i9001", 1200), ctx("cus_ivo09"), injection_suspected=False
    )
    assert out.status is ActionStatus.BLOCKED and out.eligibility
    assert out.eligibility.reasons[:2] == [
        RefundReason.ACCOUNT_INACTIVE,
        RefundReason.OUTSIDE_WINDOW,
    ]
    assert REFUND_OUTCOME[out.eligibility.reasons[0]] is EscalationQueue.TIER2


async def test_retry_after_a_lost_response_succeeds() -> None:
    gw = gateway()
    ex = executor(gw)
    pending = await ex.propose(_refund(amount=11800), ctx("cus_hal08"), injection_suspected=False)
    assert pending.approval_id
    ex.approvals.decide(
        ApprovalDecision(
            approval_id=pending.approval_id, approved=True, reviewer="s", decided_at=seed_as_of()
        )
    )
    stored = ex.approvals.get(pending.approval_id)
    assert stored
    # The refund commits, but the executor never sees the response.
    signed: JsonObject = {
        **stored.arguments,
        "idempotency_key": stored.idempotency_key,
        "approval_id": stored.approval_id,
        "approval_token": approval_token(
            TEST_APPROVAL_KEY,
            approval_id=stored.approval_id,
            order_id="ord_h8001",
            amount_cents=11800,
            idempotency_key=stored.idempotency_key,
        ),
    }
    committed = await gw.call(ToolName.ISSUE_REFUND, signed, ctx("cus_hal08", "human"))
    assert not committed.is_error
    foreign = await gw.call(ToolName.ISSUE_REFUND, signed, ctx("cus_ava01", "human"))
    assert foreign.error and foreign.error.category is ErrorCategory.NOT_FOUND
    # The reviewer re-sends the same approval through the API/CLI path.
    retried = await ex.decide(
        ApprovalDecision(
            approval_id=pending.approval_id, approved=True, reviewer="s", decided_at=seed_as_of()
        )
    )
    assert retried.status is ActionStatus.EXECUTED and retried.result == committed
    assert len(gw.refunds) == len(seed_world().refunds) + 1
    with pytest.raises(ValueError, match="already"):
        await ex.decide(
            ApprovalDecision(
                approval_id=pending.approval_id,
                approved=False,
                reviewer="s",
                decided_at=seed_as_of(),
            )
        )
    # Re-proposing the executed full refund reports the stored outcome, not a refusal.
    again = await ex.propose(_refund(amount=11800), ctx("cus_hal08"), injection_suspected=False)
    assert again.status is ActionStatus.SKIPPED and again.approval_id == pending.approval_id


async def test_reproposing_keeps_the_stored_decision() -> None:
    ex = executor()
    first = await ex.propose(_refund(), ctx("cus_hal08"), injection_suspected=False)
    assert first.approval_id
    await ex.decide(
        ApprovalDecision(
            approval_id=first.approval_id, approved=False, reviewer="s", decided_at=seed_as_of()
        )
    )
    again = await ex.propose(_refund(), ctx("cus_hal08"), injection_suspected=False)
    assert again.status is ActionStatus.BLOCKED
    stored = ex.approvals.get(first.approval_id)
    assert stored and stored.status is ApprovalStatus.REJECTED


async def test_rejected_approval_never_runs() -> None:
    ex = executor()
    proposal = ProposedAction(
        tool=ToolName.ISSUE_REFUND,
        arguments={"order_id": "ord_h8001", "amount_cents": 100, "reason": "x"},
        rationale="x",
    )
    pending = await ex.propose(proposal, ctx("cus_hal08"), injection_suspected=False)
    assert pending.approval_id
    out = await ex.decide(
        ApprovalDecision(
            approval_id=pending.approval_id, approved=False, reviewer="s", decided_at=seed_as_of()
        )
    )
    assert out.status is ActionStatus.SKIPPED and ex.gateway.refunds == seed_world().refunds
    with pytest.raises(KeyError):
        await ex.execute_approved("apr_missing")
