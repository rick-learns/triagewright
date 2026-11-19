"""The resolver, subagent isolation, citation checks, and PII redaction."""

from __future__ import annotations

import json

import pytest

from tests.agent.helpers import (
    SHIPPING,
    TIPS,
    RedactingGuardrails,
    call,
    finalize,
    request_text,
    review,
    stack,
    submit,
    tool_results,
    triage,
)
from triagewright.agent import Coordinator, prompts
from triagewright.agent.coordinator import PURPOSES
from triagewright.agent.subagents import ACTION_TOOLS
from triagewright.contracts.agents import (
    AnswerInput,
    ResolutionStatus,
)
from triagewright.contracts.ports import TicketResolver
from triagewright.contracts.telemetry import SpanKind
from triagewright.contracts.tools import ToolName
from triagewright.testing import (
    FakeCaseRuntime,
    FakeCaseRuntimeFactory,
    NullTelemetryFactory,
    PassthroughGuardrails,
    ScriptedModelClient,
    sample_ticket,
    seed_as_of,
    seed_world,
)

pytestmark = pytest.mark.anyio


# ---- the resolver ------------------------------------------------------------------


def test_coordinator_is_a_ticket_resolver() -> None:
    assert isinstance(stack().coordinator, TicketResolver)


def test_every_purpose_needs_a_model() -> None:
    s = stack()
    with pytest.raises(ValueError, match="no model id"):
        Coordinator(
            model=ScriptedModelClient([]),
            gateway=s.gateway,
            guardrails=PassthroughGuardrails(),
            executor=s.coordinator._executor,
            telemetry=NullTelemetryFactory(),
            models={"coordinator": "m"},
        )


async def test_happy_path_response_is_built_by_code() -> None:
    s = stack(
        [
            triage(),
            call("get_order", {"order_id": "ord_a1002"}),
            call("search_kb", {"query": SHIPPING.text}),
            finalize(
                "Order ord_a1002 shipped with UPS.",
                citations=[(SHIPPING.chunk_id, "Standard shipping takes 3 to 7 business days")],
            ),
            review(),
        ]
    )
    response = await s.resolve(sample_ticket())
    assert response.status is ResolutionStatus.RESOLVED
    assert response.request_id.startswith("req_")
    assert response.mode == "replay"
    assert response.reply == "Order ord_a1002 shipped with UPS."
    assert [c.doc_id for c in response.citations] == ["kb_shipping"]
    assert response.citations[0].title == SHIPPING.title
    assert response.escalation is None and response.actions == []
    assert response.triage is not None and response.triage.category == "order_status"
    assert response.metrics is not None and response.metrics.tool_calls == 2
    telemetry = s.telemetry.last
    assert telemetry is not None
    kinds = {kind for _, kind in telemetry.spans}
    assert {
        SpanKind.REQUEST,
        SpanKind.AGENT_TURN,
        SpanKind.SUBAGENT,
        SpanKind.MODEL_CALL,
        SpanKind.TOOL_CALL,
        SpanKind.GUARDRAIL,
    } <= kinds
    assert len(telemetry.responses) == 5  # every model call is recorded


async def test_wires_onto_a_case_runtime() -> None:
    model = ScriptedModelClient([triage("other"), finalize(), review(True, "tier1")])

    def make(runtime: FakeCaseRuntime) -> Coordinator:
        return Coordinator(
            model=model,
            gateway=runtime.gateway,
            guardrails=PassthroughGuardrails(),
            executor=runtime.executor,
            telemetry=NullTelemetryFactory(),
            models=dict.fromkeys(PURPOSES, "test-model"),
        )

    runtime = FakeCaseRuntimeFactory(make).build(
        case_id="escal-001", world=seed_world(), as_of=seed_as_of(), faults=[]
    )
    response = await runtime.resolver.resolve(sample_ticket("I want a human."))
    assert response.status is ResolutionStatus.ESCALATED
    assert [e.ticket_id for e in runtime.ledger().escalations] == ["tkt_0001"]


# ---- isolation ---------------------------------------------------------------------


async def test_subagents_see_only_their_typed_input_and_tools() -> None:
    s = stack(
        [
            triage(),
            call("answer"),
            call("search_kb", {"query": SHIPPING.text}),
            submit(
                "answer",
                {
                    "answer": "It takes 3 to 7 business days.",
                    "citations": [
                        {
                            "doc_id": "kb_shipping",
                            "chunk_id": SHIPPING.chunk_id,
                            "title": "made up",
                            "quote": "Standard shipping takes 3 to 7 business days",
                        },
                        {
                            "doc_id": "kb_shipping",
                            "chunk_id": SHIPPING.chunk_id,
                            "title": "x",
                            "quote": "Shipping is always free",
                        },
                    ],
                    "grounded": True,
                    "confidence": 0.8,
                },
            ),
            call("action"),
            submit("action", {"summary": "Nothing to do."}),
            finalize(),
            review(),
        ]
    )
    await s.resolve(sample_ticket())
    (answer, _) = s.requests("answer")
    assert answer.system == prompts.ANSWER
    assert {t.name for t in answer.tools} == {"search_kb", "submit_answer"}
    assert len(answer.messages) == 1
    first = answer.messages[0].content[0]
    assert first.type == "text"
    payload = AnswerInput.model_validate_json(first.text.removeprefix("<data>").split("</data>")[0])
    assert payload.ticket.ticket_id == "tkt_0001"
    assert prompts.COORDINATOR not in request_text(answer)
    assert {c.caller for c in s.calls(ToolName.SEARCH_KB)} == {"answer"}
    (action,) = s.requests("action")
    assert {t.name for t in action.tools} == {t.value for t in ACTION_TOOLS} | {
        "propose_action",
        "submit_action",
    }
    # The coordinator sees the answer draft with code-checked citations only.
    content = json.loads(tool_results(s.requests("coordinator")[1])[0].content)
    assert [c["quote"] for c in content["citations"]] == [
        "Standard shipping takes 3 to 7 business days"
    ]
    assert content["citations"][0]["title"] == SHIPPING.title
    triage_request = s.requests("triage")[0]
    assert triage_request.tools[0].name == "submit_triage" and len(triage_request.tools) == 1
    reviewer = s.requests("escalation")[0]
    assert "It takes 3 to 7" not in request_text(reviewer)  # reviewer sees the final draft
    assert "Your order is on its way." in request_text(reviewer)


async def test_citations_are_checked_by_code() -> None:
    s = stack(
        [
            triage("policy_question"),
            call("search_kb", {"query": SHIPPING.text}),
            call("search_kb", {"query": TIPS.text}),
            finalize(
                "Shipping takes 3 to 7 business days.",
                citations=[
                    (SHIPPING.chunk_id, "Standard shipping takes 3 to 7 business days"),
                    (SHIPPING.chunk_id, "Standard shipping takes 3 to 7 business days"),
                    (SHIPPING.chunk_id, "Shipping takes 1 day"),
                    (TIPS.chunk_id, TIPS.text[:40]),
                    ("kb_warranty#0", "Never searched"),
                ],
            ),
            review(),
        ]
    )
    response = await s.resolve(sample_ticket("How long does shipping take?"))
    assert [(c.chunk_id, c.quote) for c in response.citations] == [
        (SHIPPING.chunk_id, "Standard shipping takes 3 to 7 business days")
    ]


# ---- PII ---------------------------------------------------------------------------


async def test_customer_text_is_redacted_before_models_and_in_the_reply() -> None:
    guard = RedactingGuardrails(
        {"4111 1111 1111 1111": "[CARD_1]", "ava.l@example.com": "[EMAIL_1]"}
    )
    s = stack([triage(), finalize("We'll write to ava.l@example.com."), review()], guardrails=guard)
    ticket = sample_ticket("My card 4111 1111 1111 1111 was charged. Email ava.l@example.com")
    response = await s.resolve(ticket)
    assert all("4111" not in request_text(r) for r in s.requests())
    assert "[CARD_1]" in request_text(s.requests("triage")[0])
    assert response.reply == "We'll write to [EMAIL_1]."
