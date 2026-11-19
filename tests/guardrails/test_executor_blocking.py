"""The guarded executor blocks unsafe proposals, even with a buggy guardrail."""

from __future__ import annotations

import pytest

from tests.guardrails.executor_helpers import calls, ctx, executor, gateway, refund
from triagewright.contracts.agents import ActionStatus, ProposedAction
from triagewright.contracts.guardrails import (
    ContentSource,
    GuardrailDecision,
    GuardrailVerdict,
    PolicyConfig,
    PolicyContext,
    RedactionResult,
)
from triagewright.contracts.policy import REFUND_OUTCOME
from triagewright.contracts.tools import ErrorCategory, RefundReason, ToolName
from triagewright.contracts.world import EscalationQueue
from triagewright.guardrails import GuardedActionExecutor, SqliteApprovalStore
from triagewright.testing import TEST_APPROVAL_KEY, FakeToolGateway, seed_as_of, seed_world

pytestmark = pytest.mark.anyio


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


async def test_policy_limits_come_from_the_gateway() -> None:
    gw = FakeToolGateway(
        seed_world(), as_of=seed_as_of(), policy=PolicyConfig(refund_hard_limit_cents=100_000)
    )
    ex, _, _ = executor(gw)
    out = await ex.propose(refund("ord_c3001", 89900), ctx("cus_cho03"), injection_suspected=False)
    assert out.status is ActionStatus.PENDING_APPROVAL and out.eligibility
    assert out.eligibility.eligible


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
