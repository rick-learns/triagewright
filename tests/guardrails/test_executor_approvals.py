"""The guarded executor's approval lifecycle: propose, decide, execute once."""

from __future__ import annotations

from datetime import datetime

import pytest

from tests.guardrails.executor_helpers import (
    approve,
    calls,
    ctx,
    executor,
    fault,
    gateway,
    new_refunds,
    refund,
)
from triagewright.contracts.agents import ActionStatus
from triagewright.contracts.common import JsonObject
from triagewright.contracts.guardrails import ApprovalStatus
from triagewright.contracts.policy import approval_token, refund_idempotency_key
from triagewright.contracts.ports import ActionExecutor
from triagewright.contracts.tools import ErrorCategory, ToolName
from triagewright.guardrails import ApprovalConflictError, GuardedActionExecutor, RuleGuardrails
from triagewright.testing import TEST_APPROVAL_KEY, InMemoryApprovalStore, seed_as_of

pytestmark = pytest.mark.anyio


async def test_satisfies_the_port() -> None:
    assert isinstance(executor()[0], ActionExecutor)


async def test_refund_goes_to_approval_and_runs_once_after_a_human_approves() -> None:
    ex, gw, store = executor()
    pending = await ex.propose(refund(), ctx(), injection_suspected=False)
    assert pending.status is ActionStatus.PENDING_APPROVAL and pending.approval_id
    assert pending.verdict and pending.verdict.rule_ids == ["approval.irreversible"]
    assert pending.eligibility and pending.eligibility.eligible
    assert new_refunds(gw) == 0 and calls(gw, ToolName.ISSUE_REFUND) == []
    stored = store.get(pending.approval_id)
    assert stored and stored.idempotency_key == refund_idempotency_key(pending.approval_id)
    assert stored.arguments == refund().arguments and stored.requested_at == seed_as_of()
    assert store.list_pending() == [stored]

    not_yet = await ex.execute_approved(pending.approval_id)
    assert not_yet.status is ActionStatus.BLOCKED and calls(gw, ToolName.ISSUE_REFUND) == []

    done = await ex.decide(approve(pending.approval_id))
    assert done.status is ActionStatus.EXECUTED and done.attempts == 1
    assert new_refunds(gw) == 1 and gw.refunds[-1].approval_id == pending.approval_id
    assert calls(gw, ToolName.CHECK_REFUND_ELIGIBILITY) == ["action"]
    assert calls(gw, ToolName.ISSUE_REFUND) == ["human"]
    stored = store.get(pending.approval_id)
    assert stored and stored.status is ApprovalStatus.EXECUTED
    assert (await ex.execute_approved(pending.approval_id)).status is ActionStatus.BLOCKED


async def test_execution_signs_only_the_stored_arguments() -> None:
    ex, gw, store = executor()
    pending = await ex.propose(refund(), ctx(), injection_suspected=False)
    assert pending.approval_id
    await ex.decide(approve(pending.approval_id))
    sent = next(c for c in gw.calls if c.tool is ToolName.ISSUE_REFUND)
    stored = store.get(pending.approval_id)
    assert stored
    assert sent.arguments == {
        **stored.arguments,
        "idempotency_key": refund_idempotency_key(pending.approval_id),
        "approval_id": pending.approval_id,
        "approval_token": approval_token(
            TEST_APPROVAL_KEY,
            approval_id=pending.approval_id,
            order_id="ord_h8001",
            amount_cents=5900,
            idempotency_key=refund_idempotency_key(pending.approval_id),
        ),
    }


async def test_a_wrong_key_is_refused_by_the_gateway() -> None:
    ex, gw, _ = executor()
    ex._key = b"another-key-another-key-another-key"
    pending = await ex.propose(refund(), ctx(), injection_suspected=False)
    assert pending.approval_id
    out = await ex.decide(approve(pending.approval_id))
    assert out.status is ActionStatus.FAILED and out.result and out.result.error
    assert out.result.error.category is ErrorCategory.PERMISSION and new_refunds(gw) == 0


async def test_a_failed_execution_stays_approved_and_retries_with_the_same_key() -> None:
    ex, gw, store = executor(gateway([fault(ToolName.ISSUE_REFUND, ErrorCategory.TIMEOUT)]))
    pending = await ex.propose(refund(), ctx(), injection_suspected=False)
    assert pending.approval_id
    failed = await ex.decide(approve(pending.approval_id))
    assert failed.status is ActionStatus.FAILED and failed.result and failed.result.error
    stored = store.get(pending.approval_id)
    assert stored and stored.status is ApprovalStatus.APPROVED
    retried = await ex.execute_approved(pending.approval_id)
    assert retried.status is ActionStatus.EXECUTED and new_refunds(gw) == 1
    keys = [c.arguments["idempotency_key"] for c in gw.calls if c.tool is ToolName.ISSUE_REFUND]
    assert keys == [refund_idempotency_key(pending.approval_id)] * 2


async def test_retry_after_a_lost_response_succeeds_through_decide() -> None:
    ex, gw, store = executor()
    pending = await ex.propose(refund(amount=11800), ctx(), injection_suspected=False)
    assert pending.approval_id
    store.decide(approve(pending.approval_id))
    stored = store.get(pending.approval_id)
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
    committed = await gw.call(ToolName.ISSUE_REFUND, signed, ctx(caller="human"))
    assert not committed.is_error
    # The reviewer re-sends the approval; the full balance is now spent, yet it succeeds.
    retried = await ex.decide(approve(pending.approval_id))
    assert retried.status is ActionStatus.EXECUTED and retried.result == committed
    assert new_refunds(gw) == 1
    stored = store.get(pending.approval_id)
    assert stored and stored.status is ApprovalStatus.EXECUTED


async def test_repeat_decisions_on_an_executed_request_are_conflicts() -> None:
    ex, gw, _ = executor()
    pending = await ex.propose(refund(), ctx(), injection_suspected=False)
    assert pending.approval_id
    await ex.decide(approve(pending.approval_id))
    for approved in (True, False):
        with pytest.raises(ApprovalConflictError, match="already executed") as exc:
            await ex.decide(approve(pending.approval_id, approved))
        assert exc.value.status is ApprovalStatus.EXECUTED
        assert isinstance(exc.value, ValueError)  # the port's documented type
    assert new_refunds(gw) == 1 and len(calls(gw, ToolName.ISSUE_REFUND)) == 1


async def test_rejected_requests_never_run_and_reject_repeats() -> None:
    ex, gw, _ = executor()
    pending = await ex.propose(refund(), ctx(), injection_suspected=False)
    assert pending.approval_id
    out = await ex.decide(approve(pending.approval_id, approved=False))
    assert out.status is ActionStatus.SKIPPED and out.approval_id == pending.approval_id
    for approved in (True, False):
        with pytest.raises(ApprovalConflictError) as exc:
            await ex.decide(approve(pending.approval_id, approved))
        assert exc.value.status is ApprovalStatus.REJECTED
    assert (await ex.execute_approved(pending.approval_id)).status is ActionStatus.BLOCKED
    assert calls(gw, ToolName.ISSUE_REFUND) == []
    with pytest.raises(KeyError):
        await ex.decide(approve("apr_missing1"))
    with pytest.raises(KeyError):
        await ex.execute_approved("apr_missing1")


async def test_reproposing_reports_the_stored_status_without_rechecking() -> None:
    ex, gw, store = executor()
    first = await ex.propose(refund(amount=11800), ctx(), injection_suspected=False)
    assert first.approval_id
    again = await ex.propose(refund(amount=11800), ctx(), injection_suspected=False)
    assert again.status is ActionStatus.PENDING_APPROVAL and again.approval_id == first.approval_id
    await ex.decide(approve(first.approval_id))
    lookups = len(calls(gw, ToolName.CHECK_REFUND_ELIGIBILITY))
    # The full balance is spent now, but the executed approval is reported, not a refusal.
    executed = await ex.propose(refund(amount=11800), ctx(), injection_suspected=False)
    assert executed.status is ActionStatus.SKIPPED and executed.approval_id == first.approval_id
    assert len(calls(gw, ToolName.CHECK_REFUND_ELIGIBILITY)) == lookups

    rejected_first = await ex.propose(refund("ord_h8002", 100), ctx(), injection_suspected=False)
    assert rejected_first.approval_id
    await ex.decide(approve(rejected_first.approval_id, approved=False))
    rejected = await ex.propose(refund("ord_h8002", 100), ctx(), injection_suspected=False)
    assert rejected.status is ActionStatus.BLOCKED
    assert rejected.approval_id == rejected_first.approval_id
    stored = store.get(rejected_first.approval_id)
    assert stored and stored.status is ApprovalStatus.REJECTED
    # Injection still wins over a stored approval.
    flagged = await ex.propose(refund("ord_h8002", 100), ctx(), injection_suspected=True)
    assert flagged.verdict and flagged.verdict.rule_ids == ["injection.write_blocked"]


async def test_stored_approval_for_another_customer_is_blocked() -> None:
    ex, _, _ = executor()
    pending = await ex.propose(refund(), ctx(), injection_suspected=False)
    assert pending.approval_id
    other = await ex.propose(refund(), ctx("cus_ava01"), injection_suspected=False)
    assert other.status is ActionStatus.BLOCKED and other.verdict
    assert other.verdict.rule_ids == ["scope.customer_mismatch"]


async def test_works_with_the_in_memory_store_too() -> None:
    gw = gateway()
    ex = GuardedActionExecutor(
        gw,
        RuleGuardrails(),
        InMemoryApprovalStore(),
        approval_key=TEST_APPROVAL_KEY,
        clock=seed_as_of,
    )
    pending = await ex.propose(refund(), ctx(), injection_suspected=False)
    assert pending.approval_id
    assert (await ex.decide(approve(pending.approval_id))).status is ActionStatus.EXECUTED
    with pytest.raises(ApprovalConflictError):
        await ex.decide(approve(pending.approval_id))


async def test_clock_is_read_once_per_proposal() -> None:
    times = iter([datetime.fromisoformat("2026-06-15T12:00:00Z")])
    ex, _, store = executor()
    ex._clock = lambda: next(times)
    pending = await ex.propose(refund(), ctx(), injection_suspected=False)
    assert pending.approval_id
    stored = store.get(pending.approval_id)
    assert stored and stored.requested_at == seed_as_of()
