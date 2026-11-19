from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError
from scripts.export_schemas import main as export_schemas

from triagewright.contracts.agents import FinalResponse, ResolutionStatus, TriageResult
from triagewright.contracts.evals import EvalCase
from triagewright.contracts.llm import (
    Message,
    ModelRequest,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from triagewright.contracts.tools import (
    IRREVERSIBLE_TOOLS,
    TOOL_SPECS,
    ErrorCategory,
    GetOrderInput,
    ToolName,
    ToolResult,
)
from triagewright.contracts.world import Order, OrderItem, OrderStatus

NOW = datetime(2026, 6, 15, tzinfo=UTC)


def test_committed_schemas_are_current() -> None:
    assert export_schemas(["--check"]) == 0, "run: uv run python scripts/export_schemas.py"


def test_every_tool_has_a_spec_and_only_refunds_are_irreversible() -> None:
    assert set(TOOL_SPECS) == set(ToolName)
    assert frozenset({ToolName.ISSUE_REFUND}) == IRREVERSIBLE_TOOLS


def test_tool_result_envelope_requires_exactly_one_of_data_or_error() -> None:
    ok = ToolResult.ok(ToolName.GET_ORDER, GetOrderInput(order_id="ord_a1001"))
    assert not ok.is_error and ok.data == {"order_id": "ord_a1001"}
    with pytest.raises(ValidationError):
        ToolResult(tool=ToolName.GET_ORDER, is_error=True, data={"x": 1})
    with pytest.raises(ValidationError):
        ToolResult(tool=ToolName.GET_ORDER, is_error=False)


@pytest.mark.parametrize(
    ("category", "retryable"),
    [
        (ErrorCategory.TRANSIENT, True),
        (ErrorCategory.RATE_LIMITED, True),
        (ErrorCategory.TIMEOUT, True),
        (ErrorCategory.NOT_FOUND, False),
        (ErrorCategory.POLICY, False),
        (ErrorCategory.VALIDATION, False),
    ],
)
def test_retryable_defaults_follow_category(category: ErrorCategory, retryable: bool) -> None:
    result = ToolResult.fail(ToolName.GET_ORDER, category, "boom")
    assert result.error is not None and result.error.retryable is retryable


def test_retryable_can_be_overridden() -> None:
    result = ToolResult.fail(ToolName.GET_ORDER, ErrorCategory.INTERNAL, "x", retryable=True)
    assert result.error is not None and result.error.retryable


def test_contracts_are_frozen_and_reject_unknown_fields() -> None:
    order_input = GetOrderInput(order_id="ord_a1001")
    with pytest.raises(ValidationError):
        order_input.order_id = "ord_b2001"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        GetOrderInput.model_validate({"order_id": "ord_a1001", "extra": 1})


def test_ids_are_validated() -> None:
    with pytest.raises(ValidationError):
        GetOrderInput(order_id="12345")


def test_order_total_must_match_items() -> None:
    item = OrderItem(sku="X", name="X", quantity=2, unit_price_cents=500)
    Order(
        order_id="ord_t001",
        customer_id="cus_t001",
        status=OrderStatus.PLACED,
        items=[item],
        total_cents=1000,
        placed_at=NOW,
    )
    with pytest.raises(ValidationError, match="total_cents"):
        Order(
            order_id="ord_t001",
            customer_id="cus_t001",
            status=OrderStatus.PLACED,
            items=[item],
            total_cents=999,
            placed_at=NOW,
        )


def test_content_blocks_round_trip_through_json() -> None:
    message = Message(
        role="assistant",
        content=[
            TextBlock(text="checking"),
            ToolUseBlock(id="toolu_1", name="get_order", input={"order_id": "ord_a1001"}),
        ],
    )
    reply = Message(role="user", content=[ToolResultBlock(tool_use_id="toolu_1", content="{}")])
    for msg in (message, reply):
        assert Message.model_validate_json(msg.model_dump_json()) == msg


def test_fingerprint_ignores_purpose_but_not_content() -> None:
    base = ModelRequest(
        model="m", system="s", messages=[Message(role="user", content=[TextBlock(text="hi")])]
    )
    assert base.fingerprint() == base.model_copy(update={"purpose": "triage"}).fingerprint()
    assert base.fingerprint() != base.model_copy(update={"system": "other"}).fingerprint()


def test_triage_confidence_is_bounded() -> None:
    with pytest.raises(ValidationError):
        TriageResult.model_validate(
            {
                "category": "other",
                "intent": "x",
                "urgency": "low",
                "needs_human": False,
                "confidence": 1.5,
            }
        )


def test_final_response_minimal() -> None:
    resp = FinalResponse(
        request_id="req_00000001",
        ticket_id="tkt_0001",
        status=ResolutionStatus.NEEDS_INFO,
        reply="Which order is this about?",
    )
    assert resp.mode == "stub"


def test_eval_case_id_must_match_category_prefix_pattern() -> None:
    raw = {
        "id": "happy-001",
        "category": "happy_path",
        "title": "t",
        "ticket": {"ticket_id": "tkt_0001", "body": "hi", "created_at": NOW.isoformat()},
        "expected": {"status_any": ["resolved"], "escalate": False},
    }
    EvalCase.model_validate(raw)
    with pytest.raises(ValidationError):
        EvalCase.model_validate({**raw, "id": "case-1"})
