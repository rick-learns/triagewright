from __future__ import annotations

from collections.abc import Sequence

import pytest

from triagewright.contracts.agents import ProposedAction, ResolutionStatus
from triagewright.contracts.common import JsonObject
from triagewright.contracts.evals import ToolFault
from triagewright.contracts.guardrails import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalStatus,
    ContentSource,
    GuardrailDecision,
    PolicyContext,
)
from triagewright.contracts.ports import (
    ApprovalStore,
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
from triagewright.contracts.tools import CallContext, ErrorCategory, ToolError, ToolName
from triagewright.testing import (
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
CTX = CallContext(request_id="req_00000001", ticket_id="tkt_0001", caller="action")


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


def test_fakes_satisfy_ports() -> None:
    assert isinstance(gateway(), ToolGateway)
    assert isinstance(FakeRetriever(seed_kb()), Retriever)
    assert isinstance(PassthroughGuardrails(), Guardrails)
    assert isinstance(InMemoryApprovalStore(), ApprovalStore)
    assert isinstance(ScriptedModelClient([]), ModelClient)
    assert isinstance(NullTelemetry(), RequestTelemetry)
    assert isinstance(NullTelemetryFactory(), TelemetryFactory)
    assert isinstance(StubResolver(), TicketResolver)


async def test_lookups_and_not_found() -> None:
    gw = gateway()
    order = await gw.call(ToolName.GET_ORDER, {"order_id": "ord_a1001"}, CTX)
    assert not order.is_error and order.data and order.data["status"] == "delivered"
    missing = await gw.call(ToolName.GET_ORDER, {"order_id": "ord_zzzz"}, CTX)
    assert missing.error and missing.error.category is ErrorCategory.NOT_FOUND
    assert not missing.error.retryable
    lookups: list[tuple[ToolName, JsonObject]] = [
        (ToolName.GET_CUSTOMER, {"customer_id": "cus_ava01"}),
        (ToolName.GET_ACCOUNT, {"customer_id": "cus_dev04"}),
        (ToolName.LIST_ORDERS, {"customer_id": "cus_hal08", "limit": 2}),
    ]
    for name, args in lookups:
        result = await gw.call(name, args, CTX)
        assert not result.is_error, result
    listing = await gw.call(ToolName.LIST_ORDERS, {"customer_id": "cus_nope1"}, CTX)
    assert listing.is_error


async def test_validation_error_for_bad_arguments() -> None:
    result = await gateway().call(ToolName.GET_ORDER, {"order_id": 7}, CTX)
    assert result.error and result.error.category is ErrorCategory.VALIDATION


async def test_refund_eligibility_rules() -> None:
    gw = gateway()

    async def check(order_id: str, amount: int) -> JsonObject:
        r = await gw.call(
            ToolName.CHECK_REFUND_ELIGIBILITY, {"order_id": order_id, "amount_cents": amount}, CTX
        )
        assert r.data is not None
        return r.data

    assert (await check("ord_h8001", 5000))["eligible"] is True
    assert (await check("ord_b2001", 1000))["eligible"] is False  # delivered 44 days ago
    assert (await check("ord_a1002", 100))["eligible"] is False  # not delivered
    partial = await check("ord_a1001", 12900)  # 3000 already refunded
    assert partial["eligible"] is False and partial["max_refundable_cents"] == 9900


async def test_issue_refund_requires_approval_and_is_idempotent() -> None:
    gw = gateway()
    args: JsonObject = {
        "order_id": "ord_h8001",
        "amount_cents": 5900,
        "reason": "damaged",
        "idempotency_key": "tkt_0001-refund",
    }
    denied = await gw.call(ToolName.ISSUE_REFUND, args, CTX)
    assert denied.error and denied.error.category is ErrorCategory.PERMISSION
    approved: JsonObject = {**args, "approval_id": "apr_0001"}
    first = await gw.call(ToolName.ISSUE_REFUND, approved, CTX)
    again = await gw.call(ToolName.ISSUE_REFUND, approved, CTX)
    assert first == again and not first.is_error
    conflict = await gw.call(
        ToolName.ISSUE_REFUND,
        {**approved, "amount_cents": 100},
        CTX,
    )
    assert conflict.error and conflict.error.category is ErrorCategory.CONFLICT
    too_old = await gw.call(
        ToolName.ISSUE_REFUND,
        {**approved, "order_id": "ord_b2001", "idempotency_key": "tkt_0001-other"},
        CTX,
    )
    assert too_old.error and too_old.error.category is ErrorCategory.POLICY


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


async def test_search_kb_and_escalation() -> None:
    gw = gateway()
    hits = await gw.call(
        ToolName.SEARCH_KB, {"query": "refund window 30 days", "categories": ["refunds"]}, CTX
    )
    assert hits.data and hits.data["hits"]
    esc = await gw.call(
        ToolName.CREATE_ESCALATION,
        {
            "ticket_id": "tkt_0001",
            "queue": "billing",
            "priority": "high",
            "summary": "Refund over limit",
            "reason_codes": ["policy_limit"],
        },
        CTX,
    )
    assert esc.data and esc.data["escalation_id"] == "esc_0001"
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
    ctx = PolicyContext(ticket_id="tkt_0001", now=seed_as_of())
    refund = ProposedAction(tool=ToolName.ISSUE_REFUND, arguments={}, rationale="x")
    lookup = ProposedAction(tool=ToolName.GET_ORDER, arguments={}, rationale="x")
    assert g.check_action(refund, ctx).decision is GuardrailDecision.REQUIRE_APPROVAL
    assert g.check_action(lookup, ctx).decision is GuardrailDecision.ALLOW
    assert g.redact("a@b.com").text == "a@b.com"
    assert g.screen("hi", ContentSource.CUSTOMER).decision is GuardrailDecision.ALLOW


def test_approval_store_lifecycle() -> None:
    store = InMemoryApprovalStore()
    req = ApprovalRequest(
        approval_id="apr_0001",
        ticket_id="tkt_0001",
        tool=ToolName.ISSUE_REFUND,
        arguments={"order_id": "ord_h8001", "amount_cents": 5900},
        rationale="damaged",
        verdict=PassthroughGuardrails().check_action(
            ProposedAction(tool=ToolName.ISSUE_REFUND, arguments={}, rationale="x"),
            PolicyContext(ticket_id="tkt_0001", now=seed_as_of()),
        ),
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
    telemetry.record_model_call(req, resp)
    from triagewright.contracts.tools import ToolResult

    telemetry.record_tool_call(ToolResult.fail(ToolName.GET_ORDER, ErrorCategory.TIMEOUT, "t"), 5)
    metrics = telemetry.finish()
    assert metrics.input_tokens == 100 and metrics.tool_calls == 1
    assert telemetry.spans == [("turn", SpanKind.AGENT_TURN)]


async def test_stub_resolver_escalates() -> None:
    resp = await StubResolver().resolve(sample_ticket())
    assert resp.status is ResolutionStatus.ESCALATED and resp.escalation is not None
