"""The guarded executor's tool calls: eligibility retries, backoff, and reads."""

from __future__ import annotations

import pytest

from tests.guardrails.executor_helpers import calls, ctx, executor, fault, gateway, refund
from triagewright.contracts.agents import ActionStatus, LoopConfig, ProposedAction
from triagewright.contracts.guardrails import GuardrailDecision
from triagewright.contracts.tools import ErrorCategory, ToolName
from triagewright.guardrails import GuardedActionExecutor, RuleGuardrails, SqliteApprovalStore
from triagewright.testing import TEST_APPROVAL_KEY, seed_as_of

pytestmark = pytest.mark.anyio


async def test_eligibility_lookup_retries_then_reports_the_error() -> None:
    transient = ToolName.CHECK_REFUND_ELIGIBILITY
    ex, gw, _ = executor(gateway([fault(transient, ErrorCategory.TRANSIENT, times=2)]))
    ok = await ex.propose(refund(), ctx(), injection_suspected=False)
    assert ok.status is ActionStatus.PENDING_APPROVAL
    assert calls(gw, transient) == ["action"] * 3

    ex, gw, store = executor(gateway([fault(transient, ErrorCategory.TIMEOUT, times=3)]))
    failed = await ex.propose(refund(), ctx(), injection_suspected=False)
    assert failed.status is ActionStatus.FAILED and failed.attempts == 3
    assert failed.result and failed.result.error and failed.result.error.retryable
    assert failed.eligibility is None and store.list_pending() == []

    ex, gw, _ = executor()
    missing = await ex.propose(refund("ord_zzzz"), ctx(), injection_suspected=False)
    assert missing.status is ActionStatus.FAILED and missing.attempts == 1
    assert missing.result and missing.result.error
    assert missing.result.error.category is ErrorCategory.NOT_FOUND
    foreign = await ex.propose(refund("ord_c3001", 100), ctx(), injection_suspected=False)
    assert foreign.status is ActionStatus.FAILED  # another customer's order looks missing


async def test_backoff_uses_retry_after_then_the_loop_config() -> None:
    slept: list[float] = []

    async def sleep(seconds: float) -> None:
        slept.append(seconds)

    rate_limited = fault(
        ToolName.CHECK_REFUND_ELIGIBILITY, ErrorCategory.RATE_LIMITED, retry_after_ms=250
    )
    transient = fault(ToolName.CHECK_REFUND_ELIGIBILITY, ErrorCategory.TRANSIENT)
    gw = gateway([rate_limited, transient])
    ex = GuardedActionExecutor(
        gw,
        RuleGuardrails(),
        SqliteApprovalStore(),
        approval_key=TEST_APPROVAL_KEY,
        clock=seed_as_of,
        loop=LoopConfig(tool_retries=2, retry_backoff_s=0.5),
        sleep=sleep,
    )
    out = await ex.propose(refund(), ctx(), injection_suspected=False)
    assert out.status is ActionStatus.PENDING_APPROVAL and slept == [0.25, 1.0]


async def test_lookup_runs_as_the_action_caller() -> None:
    ex, gw, _ = executor()
    await ex.propose(refund(), ctx(caller="coordinator"), injection_suspected=False)
    assert calls(gw, ToolName.CHECK_REFUND_ELIGIBILITY) == ["action"]


async def test_allowed_reads_run_once() -> None:
    ex, gw, _ = executor(gateway([fault(ToolName.GET_ORDER, ErrorCategory.NOT_FOUND)]))
    read = ProposedAction(
        tool=ToolName.GET_ORDER, arguments={"order_id": "ord_h8001"}, rationale="x"
    )
    failed = await ex.propose(read, ctx(), injection_suspected=True)
    assert failed.status is ActionStatus.FAILED and failed.attempts == 1
    ok = await ex.propose(read, ctx(caller="triage"), injection_suspected=True)
    assert ok.status is ActionStatus.EXECUTED and ok.result and ok.result.data
    assert calls(gw, ToolName.GET_ORDER) == ["action", "triage"]  # the caller's own context
    assert ok.verdict and ok.verdict.decision is GuardrailDecision.ALLOW
