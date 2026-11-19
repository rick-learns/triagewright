from __future__ import annotations

from collections.abc import Sequence

import pytest

from triagewright.contracts.agents import ActionStatus, ProposedAction, ResolutionStatus
from triagewright.contracts.common import JsonObject
from triagewright.contracts.evals import ToolFault
from triagewright.contracts.guardrails import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalStatus,
    ContentSource,
    GuardrailDecision,
    GuardrailVerdict,
    PolicyContext,
)
from triagewright.contracts.policy import REFUND_OUTCOME, approval_token, refund_idempotency_key
from triagewright.contracts.ports import (
    ActionExecutor,
    ApprovalStore,
    CaseRuntime,
    CaseRuntimeFactory,
    Guardrails,
    ModelClient,
    RequestTelemetry,
    Retriever,
    TelemetryFactory,
    TicketResolver,
    ToolGateway,
)
from triagewright.contracts.rag import MetadataFilter, RetrievalQuery
from triagewright.contracts.telemetry import SpanKind
from triagewright.contracts.tools import (
    CallContext,
    Caller,
    ErrorCategory,
    RefundReason,
    ToolError,
    ToolName,
)
from triagewright.contracts.world import EscalationQueue
from triagewright.testing import (
    TEST_APPROVAL_KEY,
    FakeActionExecutor,
    FakeCaseRuntimeFactory,
    FakeRetriever,
    FakeToolGateway,
    InMemoryApprovalStore,
    NullTelemetry,
    NullTelemetryFactory,
    PassthroughGuardrails,
    ScriptedModelClient,
    StubResolver,
    model_response,
    sample_ticket,
    seed_as_of,
    seed_kb,
    seed_world,
    text_block,
    tool_use_block,
)

pytestmark = pytest.mark.anyio


def ctx(customer_id: str | None = "cus_ava01", caller: Caller = "action") -> CallContext:
    return CallContext(
        request_id="req_00000001", ticket_id="tkt_0001", customer_id=customer_id, caller=caller
    )


CTX = ctx()


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def gateway(faults: Sequence[ToolFault] = ()) -> FakeToolGateway:
    return FakeToolGateway(
        seed_world(),
        as_of=seed_as_of(),
        retriever=FakeRetriever(seed_kb()),
        faults=faults,
    )


def executor(gw: FakeToolGateway | None = None) -> FakeActionExecutor:
    return FakeActionExecutor(
        gw or gateway(), PassthroughGuardrails(), InMemoryApprovalStore(), now=seed_as_of()
    )


def test_fakes_satisfy_ports() -> None:
    assert isinstance(gateway(), ToolGateway)
    assert isinstance(FakeRetriever(seed_kb()), Retriever)
    assert isinstance(PassthroughGuardrails(), Guardrails)
    assert isinstance(InMemoryApprovalStore(), ApprovalStore)
    assert isinstance(ScriptedModelClient([]), ModelClient)
    assert isinstance(NullTelemetry(), RequestTelemetry)
    assert isinstance(NullTelemetryFactory(), TelemetryFactory)
    assert isinstance(StubResolver(), TicketResolver)
    assert isinstance(executor(), ActionExecutor)
    factory = FakeCaseRuntimeFactory()
    assert isinstance(factory, CaseRuntimeFactory)
    runtime = factory.build(case_id="happy-001", world=seed_world(), as_of=seed_as_of(), faults=[])
    assert isinstance(runtime, CaseRuntime)


async def test_lookups_and_not_found() -> None:
    gw = gateway()
    order = await gw.call(ToolName.GET_ORDER, {"order_id": "ord_a1001"}, CTX)
    assert not order.is_error and order.data and order.data["status"] == "delivered"
    missing = await gw.call(ToolName.GET_ORDER, {"order_id": "ord_zzzz"}, CTX)
    assert missing.error and missing.error.category is ErrorCategory.NOT_FOUND
    assert not missing.error.retryable
    lookups: list[tuple[ToolName, JsonObject, str]] = [
        (ToolName.GET_CUSTOMER, {"customer_id": "cus_ava01"}, "cus_ava01"),
        (ToolName.GET_ACCOUNT, {"customer_id": "cus_dev04"}, "cus_dev04"),
        (ToolName.LIST_ORDERS, {"customer_id": "cus_hal08", "limit": 2}, "cus_hal08"),
    ]
    for name, args, owner in lookups:
        result = await gw.call(name, args, ctx(owner))
        assert not result.is_error, result
    listing = await gw.call(ToolName.LIST_ORDERS, {"customer_id": "cus_nope1"}, ctx("cus_nope1"))
    assert listing.is_error
    assert [c.tool for c in gw.calls][:2] == [ToolName.GET_ORDER, ToolName.GET_ORDER]


async def test_records_are_scoped_to_the_ticket_customer() -> None:
    gw = gateway()
    foreign = await gw.call(ToolName.GET_ORDER, {"order_id": "ord_c3001"}, CTX)
    absent = await gw.call(ToolName.GET_ORDER, {"order_id": "ord_zzzz"}, CTX)
    assert foreign.error and absent.error
    assert foreign.error.category is absent.error.category is ErrorCategory.NOT_FOUND
    other = await gw.call(ToolName.GET_CUSTOMER, {"customer_id": "cus_cho03"}, CTX)
    assert other.error and other.error.category is ErrorCategory.NOT_FOUND
    unscoped = await gw.call(ToolName.GET_ORDER, {"order_id": "ord_a1001"}, ctx(None))
    assert unscoped.error and unscoped.error.category is ErrorCategory.PERMISSION
    kb = await gw.call(ToolName.SEARCH_KB, {"query": "refund"}, ctx(None))
    assert not kb.is_error


async def test_validation_error_for_bad_arguments() -> None:
    result = await gateway().call(ToolName.GET_ORDER, {"order_id": 7}, CTX)
    assert result.error and result.error.category is ErrorCategory.VALIDATION


async def test_refund_eligibility_uses_shared_policy() -> None:
    gw = gateway()

    async def check(order_id: str, amount: int, owner: str) -> JsonObject:
        r = await gw.call(
            ToolName.CHECK_REFUND_ELIGIBILITY,
            {"order_id": order_id, "amount_cents": amount},
            ctx(owner),
        )
        assert r.data is not None, r
        return r.data

    assert (await check("ord_h8001", 5000, "cus_hal08"))["eligible"] is True
    assert (await check("ord_b2001", 1000, "cus_ben02"))["reasons"] == ["outside_window"]
    assert (await check("ord_a1002", 100, "cus_ava01"))["reasons"] == ["not_delivered"]
    partial = await check("ord_a1001", 12900, "cus_ava01")  # 3000 already refunded
    assert partial["reasons"] == ["exceeds_balance"] and partial["max_refundable_cents"] == 9900
    kai = await check("ord_k1101", 1000, "cus_kai11")  # two refunds on other orders
    assert kai["reasons"] == ["refund_count_limit"] and kai["refunds_last_30d"] == 2
    assert (await check("ord_c3001", 89900, "cus_cho03"))["reasons"] == ["over_hard_limit"]
    ivo = await check("ord_i9001", 1200, "cus_ivo09")
    assert ivo["reasons"] == ["account_inactive", "outside_window"]


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


async def test_case_runtime_is_isolated_and_observable() -> None:
    factory = FakeCaseRuntimeFactory(docs=seed_kb())
    a = factory.build(case_id="happy-001", world=seed_world(), as_of=seed_as_of(), faults=[])
    b = factory.build(case_id="happy-003", world=seed_world(), as_of=seed_as_of(), faults=[])
    await a.gateway.call(ToolName.GET_ORDER, {"order_id": "ord_a1001"}, CTX)
    assert len(a.tool_calls()) == 1 and b.tool_calls() == []
    assert factory.built == ["happy-001", "happy-003"]
    assert a.ledger().refunds == seed_world().refunds and a.pending_approvals() == []
    resp = await a.resolver.resolve(sample_ticket())
    assert resp.mode == "stub"
    wired = FakeCaseRuntimeFactory(lambda rt: StubResolver())
    assert isinstance(
        wired.build(
            case_id="happy-002", world=seed_world(), as_of=seed_as_of(), faults=[]
        ).resolver,
        StubResolver,
    )


async def test_fault_injection_fails_first_n_matching_calls() -> None:
    fault = ToolFault(
        tool=ToolName.GET_ORDER,
        error=ToolError(category=ErrorCategory.TRANSIENT, message="db down", retryable=True),
        times=2,
        match_arguments={"order_id": "ord_a1001"},
    )
    gw = gateway(faults=[fault])
    other = await gw.call(ToolName.GET_ORDER, {"order_id": "ord_a1002"}, CTX)
    assert not other.is_error
    results = [await gw.call(ToolName.GET_ORDER, {"order_id": "ord_a1001"}, CTX) for _ in range(3)]
    assert [r.is_error for r in results] == [True, True, False]
    assert results[0].error and results[0].error.retryable


async def test_search_kb_marks_untrusted_docs() -> None:
    gw = gateway()
    hits = await gw.call(
        ToolName.SEARCH_KB, {"query": "refund window 30 days", "categories": ["refunds"]}, CTX
    )
    assert hits.data and hits.data["hits"]
    tips = await gw.call(ToolName.SEARCH_KB, {"query": "community tips customers", "k": 10}, CTX)
    assert tips.data and isinstance(tips.data["hits"], list)
    trusted = {h["doc_id"]: h["trusted"] for h in tips.data["hits"] if isinstance(h, dict)}
    assert trusted.get("kb_community_tips") is False
    empty = FakeToolGateway(seed_world(), as_of=seed_as_of())
    no_kb = await empty.call(ToolName.SEARCH_KB, {"query": "refund"}, CTX)
    assert no_kb.data == {"hits": []}


async def test_fake_retriever_ranks_and_filters() -> None:
    retriever = FakeRetriever(seed_kb())
    results = await retriever.retrieve(RetrievalQuery(text="account locked password reset", k=3))
    assert results and results[0].chunk.doc_id == "kb_account_lockout"
    assert [r.rank for r in results] == list(range(1, len(results) + 1))
    filtered = await retriever.retrieve(
        RetrievalQuery(text="refund", filters=MetadataFilter(tags_any=["warranty"]))
    )
    assert all("warranty" in r.chunk.tags for r in filtered)


def test_passthrough_guardrails_gate_irreversible_tools() -> None:
    g = PassthroughGuardrails()
    pctx = PolicyContext(ticket_id="tkt_0001", customer_id="cus_ava01", now=seed_as_of())
    refund = ProposedAction(tool=ToolName.ISSUE_REFUND, arguments={}, rationale="x")
    lookup = ProposedAction(tool=ToolName.GET_ORDER, arguments={}, rationale="x")
    assert g.check_action(refund, pctx).decision is GuardrailDecision.BLOCK  # no eligibility
    assert g.check_action(lookup, pctx).decision is GuardrailDecision.ALLOW
    assert g.redact("a@b.com").text == "a@b.com"
    assert g.screen("hi", ContentSource.CUSTOMER).decision is GuardrailDecision.ALLOW


def test_approval_store_lifecycle() -> None:
    store = InMemoryApprovalStore()
    req = ApprovalRequest(
        approval_id="apr_0001",
        ticket_id="tkt_0001",
        customer_id="cus_hal08",
        tool=ToolName.ISSUE_REFUND,
        arguments={"order_id": "ord_h8001", "amount_cents": 5900, "reason": "damaged"},
        idempotency_key="refund-apr_0001",
        rationale="damaged",
        verdict=GuardrailVerdict(decision=GuardrailDecision.REQUIRE_APPROVAL),
        requested_at=seed_as_of(),
    )
    store.submit(req)
    assert store.list_pending() == [req]
    with pytest.raises(ValueError, match="not approved"):
        store.mark_executed("apr_0001")
    decided = store.decide(
        ApprovalDecision(
            approval_id="apr_0001", approved=True, reviewer="supervisor", decided_at=seed_as_of()
        )
    )
    assert decided.status is ApprovalStatus.APPROVED and store.list_pending() == []
    with pytest.raises(ValueError, match="already"):
        store.decide(
            ApprovalDecision(
                approval_id="apr_0001", approved=False, reviewer="s", decided_at=seed_as_of()
            )
        )
    assert store.mark_executed("apr_0001").status is ApprovalStatus.EXECUTED
    assert store.get("apr_0001") is not None and store.get("apr_nope") is None


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


async def test_scripted_model_client_and_telemetry() -> None:
    client = ScriptedModelClient(
        [model_response(tool_use_block("get_order", {"order_id": "ord_a1001"}))]
    )
    from triagewright.contracts.llm import Message, ModelRequest

    req = ModelRequest(
        model="m", system="s", messages=[Message(role="user", content=[text_block("hi")])]
    )
    resp = await client.complete(req)
    assert resp.stop_reason == "tool_use" and client.requests == [req]
    with pytest.raises(RuntimeError, match="ran out"):
        await client.complete(req)
    telemetry = NullTelemetryFactory().for_request("req_00000001", "tkt_0001")
    with telemetry.span("turn", SpanKind.AGENT_TURN) as span:
        span.set_attribute("turn", 1)
        span.set_error("x")
    telemetry.record_model_call(req, resp, started_ms=0.0)
    from triagewright.contracts.tools import ToolResult

    telemetry.record_tool_call(ToolResult.fail(ToolName.GET_ORDER, ErrorCategory.TIMEOUT, "t"), 5)
    metrics = telemetry.finish()
    assert metrics.input_tokens == 100 and metrics.tool_calls == 1
    assert telemetry.spans == [("turn", SpanKind.AGENT_TURN)]


async def test_stub_resolver_escalates() -> None:
    resp = await StubResolver().resolve(sample_ticket())
    assert resp.status is ResolutionStatus.ESCALATED and resp.escalation is not None
