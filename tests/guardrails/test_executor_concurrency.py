"""The guarded executor under racing reviewers and workers."""

from __future__ import annotations

import pytest

from tests.guardrails.executor_helpers import approve, calls, ctx, executor, new_refunds, refund
from triagewright.contracts.agents import ActionStatus
from triagewright.contracts.guardrails import ApprovalDecision, ApprovalStatus
from triagewright.contracts.tools import ToolName
from triagewright.guardrails import ApprovalConflictError

pytestmark = pytest.mark.anyio


async def test_concurrent_approval_resumes_instead_of_conflicting() -> None:
    ex, gw, store = executor()
    pending = await ex.propose(refund(), ctx(), injection_suspected=False)
    assert pending.approval_id
    real_decide = store.decide

    def racing_decide(decision: ApprovalDecision) -> object:
        real_decide(decision)  # another reviewer's approve lands first
        return real_decide(decision)

    store.decide = racing_decide  # type: ignore[method-assign,assignment]
    out = await ex.decide(approve(pending.approval_id))
    assert out.status is ActionStatus.EXECUTED and new_refunds(gw) == 1


async def test_concurrent_execution_is_reported_once_committed() -> None:
    ex, gw, store = executor()
    pending = await ex.propose(refund(), ctx(), injection_suspected=False)
    assert pending.approval_id
    store.decide(approve(pending.approval_id))
    real_mark = store.mark_executed

    def racing_mark(approval_id: str) -> object:
        real_mark(approval_id)  # a concurrent execute_approved recorded it first
        return real_mark(approval_id)

    store.mark_executed = racing_mark  # type: ignore[method-assign,assignment]
    out = await ex.execute_approved(pending.approval_id)
    assert out.status is ActionStatus.EXECUTED and new_refunds(gw) == 1


async def test_an_approve_racing_a_finished_execution_is_a_conflict() -> None:
    ex, gw, store = executor()
    pending = await ex.propose(refund(), ctx(), injection_suspected=False)
    assert pending.approval_id
    store.decide(approve(pending.approval_id))
    real_get = store.get
    reads = 0

    def get_then_execute_elsewhere(approval_id: str) -> object:
        nonlocal reads
        reads += 1
        if reads == 2:  # another worker executes between decide's read and execution
            store.mark_executed(approval_id)
        return real_get(approval_id)

    store.get = get_then_execute_elsewhere  # type: ignore[method-assign,assignment]
    with pytest.raises(ApprovalConflictError) as exc:
        await ex.decide(approve(pending.approval_id))
    assert exc.value.status is ApprovalStatus.EXECUTED
    assert calls(gw, ToolName.ISSUE_REFUND) == []
