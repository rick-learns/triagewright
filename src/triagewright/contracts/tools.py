"""MCP tool signatures and the structured error envelope.

Every tool returns a :class:`ToolResult`.

Scoping: every call carries a :class:`CallContext` whose ``customer_id`` comes
from the ticket, never from model arguments. Private tools (all but
:data:`UNSCOPED_TOOLS`) fail with ``permission`` when it is missing, and
records owned by another customer are reported as ``not_found``, exactly like
missing ones. ``create_escalation`` is code-owned: only ``caller="coordinator"``
may call it, and only for ``ctx.ticket_id``, so anonymous tickets can still be
handed off.
 Over MCP it travels as a
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
    RequestId,
    TicketId,
)
from triagewright.contracts.world import (
    Account,
    Customer,
    Escalation,
    EscalationQueue,
    EscalationReason,
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


class RefundProposal(Contract):
    """What the action agent may propose for ``issue_refund``. The executor adds the rest."""

    order_id: OrderId
    amount_cents: Annotated[int, Field(gt=0)]
    reason: Annotated[str, Field(min_length=1, max_length=500)]


class IssueRefundInput(RefundProposal):
    """Only the approval executor builds this, from a stored, human-approved request."""

    idempotency_key: Annotated[str, Field(min_length=8, max_length=128)]  # refund-<approval_id>
    approval_id: ApprovalId
    approval_token: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]  # policy.approval_token


class CreateEscalationInput(Contract):
    """Idempotent per ticket: a second call for the same ticket returns the first escalation."""

    ticket_id: TicketId
    queue: EscalationQueue
    priority: Priority
    summary: Annotated[str, Field(min_length=1, max_length=2000)]
    reason_codes: Annotated[list[EscalationReason], Field(min_length=1)]


# ---- tool outputs ---------------------------------------------------------------


class OrderList(Contract):
    orders: list[Order]


class KBHit(Contract):
    doc_id: DocId
    chunk_id: ChunkId
    title: str
    category: str
    trusted: bool  # False for user-submitted docs; never cite them as policy
    text: str
    score: float


class KBSearchResult(Contract):
    hits: list[KBHit]


class RefundReason(StrEnum):
    """Why a refund can't go to approval, in precedence order (first wins).

    See ``contracts.policy.REFUND_OUTCOME`` for what the agent does for each.
    """

    ACCOUNT_INACTIVE = "account_inactive"  # customer suspended or closed
    NOT_DELIVERED = "not_delivered"  # status is not delivered/returned
    OUTSIDE_WINDOW = "outside_window"  # more than window days after delivery
    EXCEEDS_BALANCE = "exceeds_balance"  # amount > total minus already refunded
    OVER_HARD_LIMIT = "over_hard_limit"  # amount > refund_hard_limit_cents
    REFUND_COUNT_LIMIT = "refund_count_limit"  # already at max refunds in 30 days


class RefundEligibility(Contract):
    """Output of ``contracts.policy.evaluate_refund``."""

    order_id: OrderId
    eligible: bool
    max_refundable_cents: Cents
    already_refunded_cents: Cents
    refunds_last_30d: Annotated[int, Field(ge=0)]  # customer-wide, issued refunds
    window_ends_at: datetime | None
    reasons: list[RefundReason] = []  # sorted by precedence


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
            "Issue a refund. Irreversible. Requires a human-approved approval_id and its token.",
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

# Tools that need no customer scope. Every other tool fails closed without ctx.customer_id.
UNSCOPED_TOOLS: frozenset[ToolName] = frozenset({ToolName.SEARCH_KB, ToolName.CREATE_ESCALATION})

# Tools only coordinator code may call. Models can never propose them.
CODE_OWNED_TOOLS: frozenset[ToolName] = frozenset({ToolName.CREATE_ESCALATION})


def proposal_model(tool: ToolName) -> type[BaseModel]:
    """The schema ``ProposedAction.arguments`` must match for ``tool``."""
    return RefundProposal if tool is ToolName.ISSUE_REFUND else TOOL_SPECS[tool].input_model


Caller = Literal["triage", "answer", "action", "escalation", "coordinator", "human"]


class CallContext(Contract):
    """Trusted per-call metadata, set by code from the ticket. Never from model output."""

    request_id: RequestId
    ticket_id: TicketId
    customer_id: CustomerId | None
    caller: Caller


class ToolCallRecord(Contract):
    """One raw gateway call, as logged by the gateway. Graders read these, not model claims."""

    tool: ToolName
    arguments: JsonObject
    caller: Caller
    result: ToolResult
    latency_ms: Annotated[float, Field(ge=0)] = 0.0
