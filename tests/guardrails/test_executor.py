from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import pytest

from triagewright.contracts.agents import ActionStatus, LoopConfig, ProposedAction
from triagewright.contracts.common import JsonObject
from triagewright.contracts.evals import ToolFault
from triagewright.contracts.guardrails import (
    ApprovalDecision,
    ApprovalStatus,
    ContentSource,
    GuardrailDecision,
    GuardrailVerdict,
    PolicyConfig,
    PolicyContext,
    RedactionResult,
)
from triagewright.contracts.policy import REFUND_OUTCOME, approval_token, refund_idempotency_key
from triagewright.contracts.ports import ActionExecutor
from triagewright.contracts.tools import (
    CallContext,
    Caller,
    ErrorCategory,
    RefundReason,
    ToolError,
    ToolName,
)
from triagewright.contracts.world import EscalationQueue
from triagewright.guardrails import (
    ApprovalConflictError,
    GuardedActionExecutor,
    RuleGuardrails,
    SqliteApprovalStore,
)
from triagewright.testing import (
    TEST_APPROVAL_KEY,
    FakeToolGateway,
    InMemoryApprovalStore,
    seed_as_of,
    seed_world,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def ctx(customer_id: str | None = "cus_hal08", caller: Caller = "action") -> CallContext:
    return CallContext(
        request_id="req_00000001", ticket_id="tkt_0001", customer_id=customer_id, caller=caller
    )


def gateway(faults: Sequence[ToolFault] = ()) -> FakeToolGateway:
    return FakeToolGateway(seed_world(), as_of=seed_as_of(), faults=faults)


def executor(
    gw: FakeToolGateway | None = None, **kwargs: object
) -> tuple[GuardedActionExecutor, FakeToolGateway, SqliteApprovalStore]:
    gw = gw or gateway()
    store = SqliteApprovalStore()
    ex = GuardedActionExecutor(
        gw,
        RuleGuardrails(),
        store,
        approval_key=TEST_APPROVAL_KEY,
        clock=seed_as_of,
        loop=LoopConfig(retry_backoff_s=0),
        **kwargs,  # type: ignore[arg-type]
    )
    return ex, gw, store


def refund(order_id: str = "ord_h8001", amount: int = 5900) -> ProposedAction:
    return ProposedAction(
        tool=ToolName.ISSUE_REFUND,
        arguments={"order_id": order_id, "amount_cents": amount, "reason": "damaged"},
        rationale="damaged on arrival",
    )


def approve(approval_id: str, approved: bool = True) -> ApprovalDecision:
    return ApprovalDecision(
        approval_id=approval_id, approved=approved, reviewer="supervisor", decided_at=seed_as_of()
    )


def fault(tool: ToolName, category: ErrorCategory, times: int = 1, **kw: int) -> ToolFault:
    error = ToolError(category=category, message="x", retryable=category.value != "not_found", **kw)
    return ToolFault(tool=tool, error=error, times=times)


def calls(gw: FakeToolGateway, tool: ToolName) -> list[Caller]:
    return [c.caller for c in gw.calls if c.tool is tool]


def new_refunds(gw: FakeToolGateway) -> int:
    return len(gw.refunds) - len(seed_world().refunds)


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


async def test_unsafe_proposals_are_blocked_before_any_call() -> None:
    ex, gw, store = executor()
    escalation = ProposedAction(
        tool=ToolName.CREATE_ESCALATION,
        arguments={
            "ticket_id": "tkt_0001",
            "queue": "security",
            "priority": "high",
            "summary": "x",
            "reason_codes": ["security"],
        },
        rationale="x",
    )
    code_owned = await ex.propose(escalation, ctx(), injection_suspected=False)
    assert code_owned.verdict and code_owned.verdict.rule_ids == ["tool.code_owned"]
    injected = await ex.propose(refund(), ctx(), injection_suspected=True)
    assert injected.status is ActionStatus.BLOCKED and injected.verdict
    assert injected.verdict.rule_ids == ["injection.write_blocked"]
    assert injected.verdict.injection_suspected
    for extra in ({"approval_id": "apr_0001"}, {"approval_token": "0" * 64}):
        smuggled = ProposedAction(
            tool=ToolName.ISSUE_REFUND, arguments={**refund().arguments, **extra}, rationale="x"
        )
        rejected = await ex.propose(smuggled, ctx(), injection_suspected=False)
        assert rejected.status is ActionStatus.BLOCKED and rejected.result
        assert rejected.result.error and rejected.result.error.category is ErrorCategory.VALIDATION
    assert gw.calls == [] and store.list_pending() == []


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


async def test_blocked_refund_carries_the_winning_reason() -> None:
    # Closed account and expired window: account_inactive wins, so tier2.
    ex, _, store = executor()
    out = await ex.propose(refund("ord_i9001", 1200), ctx("cus_ivo09"), injection_suspected=False)
    assert out.status is ActionStatus.BLOCKED and out.eligibility
    assert out.eligibility.reasons == [RefundReason.ACCOUNT_INACTIVE, RefundReason.OUTSIDE_WINDOW]
    assert out.verdict and out.verdict.rule_ids == [
        "refund.account_inactive",
        "refund.outside_window",
    ]
    assert REFUND_OUTCOME[out.eligibility.reasons[0]] is EscalationQueue.TIER2
    over = await ex.propose(refund("ord_c3001", 89900), ctx("cus_cho03"), injection_suspected=False)
    assert over.eligibility and over.eligibility.reasons == [RefundReason.OVER_HARD_LIMIT]
    assert REFUND_OUTCOME[over.eligibility.reasons[0]] is EscalationQueue.BILLING
    assert store.list_pending() == []


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


async def test_policy_limits_come_from_the_gateway() -> None:
    gw = FakeToolGateway(
        seed_world(), as_of=seed_as_of(), policy=PolicyConfig(refund_hard_limit_cents=100_000)
    )
    ex, _, _ = executor(gw)
    out = await ex.propose(refund("ord_c3001", 89900), ctx("cus_cho03"), injection_suspected=False)
    assert out.status is ActionStatus.PENDING_APPROVAL and out.eligibility
    assert out.eligibility.eligible


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


class _Permissive:
    """A buggy guardrail: allows everything, or wants approval for everything."""

    def __init__(self, decision: GuardrailDecision) -> None:
        self.decision = decision

    def redact(self, text: str) -> RedactionResult:
        return RedactionResult(text=text)

    def screen(self, text: str, source: ContentSource) -> GuardrailVerdict:
        return GuardrailVerdict(decision=GuardrailDecision.ALLOW)

    def check_action(self, action: ProposedAction, ctx: PolicyContext) -> GuardrailVerdict:
        return GuardrailVerdict(decision=self.decision)


async def test_a_guardrail_bug_never_runs_an_irreversible_tool() -> None:
    for decision in (GuardrailDecision.ALLOW, GuardrailDecision.REQUIRE_APPROVAL):
        gw = gateway()
        store = SqliteApprovalStore()
        ex = GuardedActionExecutor(
            gw, _Permissive(decision), store, approval_key=TEST_APPROVAL_KEY, clock=seed_as_of
        )
        out = await ex.propose(refund(), ctx(), injection_suspected=False)
        if decision is GuardrailDecision.ALLOW:
            assert out.status is ActionStatus.BLOCKED and out.eligibility
        else:
            assert out.status is ActionStatus.PENDING_APPROVAL
        assert calls(gw, ToolName.ISSUE_REFUND) == []
        read = ProposedAction(
            tool=ToolName.GET_ORDER, arguments={"order_id": "ord_h8001"}, rationale="x"
        )
        wants_approval = await ex.propose(read, ctx(), injection_suspected=False)
        if decision is GuardrailDecision.REQUIRE_APPROVAL:
            assert wants_approval.status is ActionStatus.BLOCKED  # reads are never stored
        else:
            assert wants_approval.status is ActionStatus.EXECUTED


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


async def test_clock_is_read_once_per_proposal() -> None:
    times = iter([datetime.fromisoformat("2026-06-15T12:00:00Z")])
    ex, _, store = executor()
    ex._clock = lambda: next(times)
    pending = await ex.propose(refund(), ctx(), injection_suspected=False)
    assert pending.approval_id
    stored = store.get(pending.approval_id)
    assert stored and stored.requested_at == seed_as_of()
