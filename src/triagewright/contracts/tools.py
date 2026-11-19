"""MCP tool signatures and the structured error envelope.

Every tool returns a :class:`ToolResult`. Over MCP it travels as a
``CallToolResult`` whose ``isError`` equals ``ToolResult.is_error``, whose
``structuredContent`` is ``ToolResult.model_dump(mode="json")``, and whose single
``TextContent`` block holds the same JSON. Tool handlers never raise to the
client: failures become ``ToolResult(is_error=True, error=ToolError(...))``.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from triagewright.contracts.common import (
    ApprovalId,
    Cents,
    ChunkId,
    Contract,
    CustomerId,
    DocId,
    JsonObject,
    OrderId,
    TicketId,
)
from triagewright.contracts.world import (
    Account,
    Customer,
    Escalation,
    EscalationQueue,
    Order,
    Priority,
    Refund,
)


class ToolName(StrEnum):
    GET_CUSTOMER = "get_customer"
    GET_ACCOUNT = "get_account"
    GET_ORDER = "get_order"
    LIST_ORDERS = "list_orders"
    SEARCH_KB = "search_kb"
    CHECK_REFUND_ELIGIBILITY = "check_refund_eligibility"
    ISSUE_REFUND = "issue_refund"
    CREATE_ESCALATION = "create_escalation"


class SideEffect(StrEnum):
    READ = "read"
    WRITE_REVERSIBLE = "write_reversible"
    WRITE_IRREVERSIBLE = "write_irreversible"


class ErrorCategory(StrEnum):
    VALIDATION = "validation"  # bad arguments; fix the call, do not retry as-is
    NOT_FOUND = "not_found"  # the referenced record does not exist
    PERMISSION = "permission"  # caller may not do this (e.g. missing approval)
    POLICY = "policy"  # business rule forbids it (e.g. outside refund window)
    CONFLICT = "conflict"  # state changed / duplicate idempotency key with other args
    RATE_LIMITED = "rate_limited"  # back off, then retry
    TRANSIENT = "transient"  # temporary backend failure, retry with backoff
    TIMEOUT = "timeout"  # backend did not answer in time
    INTERNAL = "internal"  # unexpected server bug


RETRYABLE_BY_DEFAULT: frozenset[ErrorCategory] = frozenset(
    {ErrorCategory.RATE_LIMITED, ErrorCategory.TRANSIENT, ErrorCategory.TIMEOUT}
)


class ToolError(Contract):
    category: ErrorCategory
    message: Annotated[str, Field(max_length=500)]
    retryable: bool
    retry_after_ms: Annotated[int, Field(ge=0)] | None = None
    details: dict[str, str | int | bool] = {}


class ToolResult(Contract):
    """The envelope every tool call returns. Exactly one of ``data``/``error`` is set."""

    tool: ToolName
    is_error: bool
    data: JsonObject | None = None
    error: ToolError | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> ToolResult:
        if self.is_error != (self.error is not None) or self.is_error == (self.data is not None):
            raise ValueError("is_error=True requires error only; is_error=False requires data only")
        return self

    @classmethod
    def ok(cls, tool: ToolName, data: BaseModel) -> ToolResult:
        return cls(tool=tool, is_error=False, data=data.model_dump(mode="json"))

    @classmethod
    def fail(
        cls,
        tool: ToolName,
        category: ErrorCategory,
        message: str,
        *,
        retryable: bool | None = None,
        retry_after_ms: int | None = None,
        details: dict[str, str | int | bool] | None = None,
    ) -> ToolResult:
        return cls(
            tool=tool,
            is_error=True,
            error=ToolError(
                category=category,
                message=message,
                retryable=category in RETRYABLE_BY_DEFAULT if retryable is None else retryable,
                retry_after_ms=retry_after_ms,
                details=details or {},
            ),
        )


# ---- tool inputs ----------------------------------------------------------------


class GetCustomerInput(Contract):
    customer_id: CustomerId


class GetAccountInput(Contract):
    customer_id: CustomerId


class GetOrderInput(Contract):
    order_id: OrderId


class ListOrdersInput(Contract):
    customer_id: CustomerId
    limit: Annotated[int, Field(ge=1, le=50)] = 10


class SearchKBInput(Contract):
    query: Annotated[str, Field(min_length=1, max_length=500)]
    k: Annotated[int, Field(ge=1, le=10)] = 5
    categories: list[str] | None = None


class CheckRefundEligibilityInput(Contract):
    order_id: OrderId
    amount_cents: Annotated[int, Field(gt=0)]


class IssueRefundInput(Contract):
    order_id: OrderId
    amount_cents: Annotated[int, Field(gt=0)]
    reason: Annotated[str, Field(min_length=1, max_length=500)]
    idempotency_key: Annotated[str, Field(min_length=8, max_length=128)]
    approval_id: ApprovalId | None = None


class CreateEscalationInput(Contract):
    ticket_id: TicketId
    queue: EscalationQueue
    priority: Priority
    summary: Annotated[str, Field(min_length=1, max_length=2000)]
    reason_codes: Annotated[list[str], Field(min_length=1)]


# ---- tool outputs ---------------------------------------------------------------


class OrderList(Contract):
    orders: list[Order]


class KBHit(Contract):
    doc_id: DocId
    chunk_id: ChunkId
    title: str
    text: str
    score: float


class KBSearchResult(Contract):
    hits: list[KBHit]


class RefundEligibility(Contract):
    order_id: OrderId
    eligible: bool
    max_refundable_cents: Cents
    already_refunded_cents: Cents
    window_ends_at: datetime | None
    reasons: list[str] = []


# ---- registry -------------------------------------------------------------------


class ToolSpec(Contract):
    name: ToolName
    description: str
    side_effect: SideEffect
    input_model: type[Contract]
    output_model: type[Contract]

    def input_schema(self) -> JsonObject:
        schema: JsonObject = self.input_model.model_json_schema()
        return schema


def _spec(
    name: ToolName,
    description: str,
    side_effect: SideEffect,
    input_model: type[Contract],
    output_model: type[Contract],
) -> tuple[ToolName, ToolSpec]:
    return name, ToolSpec(
        name=name,
        description=description,
        side_effect=side_effect,
        input_model=input_model,
        output_model=output_model,
    )


TOOL_SPECS: dict[ToolName, ToolSpec] = dict(
    [
        _spec(
            ToolName.GET_CUSTOMER,
            "Look up a customer profile by customer_id.",
            SideEffect.READ,
            GetCustomerInput,
            Customer,
        ),
        _spec(
            ToolName.GET_ACCOUNT,
            "Get login/security state for a customer's account.",
            SideEffect.READ,
            GetAccountInput,
            Account,
        ),
        _spec(
            ToolName.GET_ORDER,
            "Get one order with items, status, and tracking.",
            SideEffect.READ,
            GetOrderInput,
            Order,
        ),
        _spec(
            ToolName.LIST_ORDERS,
            "List a customer's most recent orders, newest first.",
            SideEffect.READ,
            ListOrdersInput,
            OrderList,
        ),
        _spec(
            ToolName.SEARCH_KB,
            "Search the support knowledge base. Returns citable chunks.",
            SideEffect.READ,
            SearchKBInput,
            KBSearchResult,
        ),
        _spec(
            ToolName.CHECK_REFUND_ELIGIBILITY,
            "Check whether an order can be refunded for an amount, and the maximum allowed.",
            SideEffect.READ,
            CheckRefundEligibilityInput,
            RefundEligibility,
        ),
        _spec(
            ToolName.ISSUE_REFUND,
            "Issue a refund. Irreversible. Requires an approved approval_id.",
            SideEffect.WRITE_IRREVERSIBLE,
            IssueRefundInput,
            Refund,
        ),
        _spec(
            ToolName.CREATE_ESCALATION,
            "Hand the ticket to a human support queue with a summary.",
            SideEffect.WRITE_REVERSIBLE,
            CreateEscalationInput,
            Escalation,
        ),
    ]
)

IRREVERSIBLE_TOOLS: frozenset[ToolName] = frozenset(
    name for name, spec in TOOL_SPECS.items() if spec.side_effect is SideEffect.WRITE_IRREVERSIBLE
)


class CallContext(Contract):
    """Per-call metadata the gateway passes alongside tool arguments."""

    request_id: str
    ticket_id: TicketId
    caller: Literal["triage", "answer", "action", "escalation", "coordinator", "human"]
