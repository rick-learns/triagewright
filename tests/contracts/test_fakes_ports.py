"""The other fakes satisfy their ports and behave as the ports promise."""

from __future__ import annotations

import pytest

from tests.contracts.fakes_helpers import CTX, executor, gateway
from triagewright.contracts.agents import ProposedAction, ResolutionStatus
from triagewright.contracts.guardrails import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalStatus,
    ContentSource,
    GuardrailDecision,
    GuardrailVerdict,
    PolicyContext,
)
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
from triagewright.contracts.tools import ErrorCategory, ToolName
from triagewright.testing import (
    FakeCaseRuntimeFactory,
    FakeRetriever,
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


async def test_fake_retriever_ranks_and_filters() -> None:
    retriever = FakeRetriever(seed_kb())
    results = await retriever.retrieve(RetrievalQuery(text="account locked password reset", k=3))
    assert results and results[0].chunk.doc_id == "kb_account_lockout"
    assert [r.rank for r in results] == list(range(1, len(results) + 1))
    text = "warranty repair defective refund"
    unfiltered = await retriever.retrieve(RetrievalQuery(text=text))
    assert any("warranty" not in r.chunk.tags for r in unfiltered)  # the filter has work to do
    filtered = await retriever.retrieve(
        RetrievalQuery(text=text, filters=MetadataFilter(tags_any=["warranty"]))
    )
    assert filtered and all("warranty" in r.chunk.tags for r in filtered)


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
