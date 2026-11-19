"""Guardrail verdicts, PII redaction, refund policy, and human approval records."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated

from pydantic import Field

from triagewright.contracts.common import ApprovalId, Contract, TicketId
from triagewright.contracts.tools import ToolName
from triagewright.contracts.world import Customer, Order, Refund


class ContentSource(StrEnum):
    CUSTOMER = "customer"
    TOOL_OUTPUT = "tool_output"
    KB_DOCUMENT = "kb_document"
    MODEL_OUTPUT = "model_output"


class PIIType(StrEnum):
    EMAIL = "email"
    PHONE = "phone"
    CARD_NUMBER = "card_number"
    SSN = "ssn"
    IP_ADDRESS = "ip_address"


class RedactionSpan(Contract):
    pii_type: PIIType
    start: Annotated[int, Field(ge=0)]  # offsets into the original text
    end: Annotated[int, Field(ge=0)]
    placeholder: str  # e.g. "[EMAIL_1]"


class RedactionResult(Contract):
    text: str
    spans: list[RedactionSpan] = []


class GuardrailDecision(StrEnum):
    ALLOW = "allow"
    BLOCK = "block"
    REQUIRE_APPROVAL = "require_approval"


class GuardrailVerdict(Contract):
    decision: GuardrailDecision
    rule_ids: list[str] = []  # stable ids, e.g. "refund.over_hard_limit", "injection.override"
    reasons: list[str] = []
    injection_suspected: bool = False


class PolicyConfig(Contract):
    refund_hard_limit_cents: Annotated[int, Field(ge=0)] = 50_000
    refund_window_days: Annotated[int, Field(ge=0)] = 30
    max_refunds_per_customer_30d: Annotated[int, Field(ge=0)] = 2
    require_approval_for_irreversible: bool = True


class PolicyContext(Contract):
    ticket_id: TicketId
    customer: Customer | None = None
    order: Order | None = None
    prior_refunds: list[Refund] = []
    injection_suspected: bool = False
    now: datetime


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"


class ApprovalRequest(Contract):
    """A proposed irreversible action waiting for a human."""

    approval_id: ApprovalId
    ticket_id: TicketId
    tool: ToolName
    arguments: dict[str, str | int | bool | None]
    rationale: str
    verdict: GuardrailVerdict
    status: ApprovalStatus = ApprovalStatus.PENDING
    requested_at: datetime


class ApprovalDecision(Contract):
    approval_id: ApprovalId
    approved: bool
    reviewer: Annotated[str, Field(min_length=1, max_length=100)]  # a role or handle, never PII
    note: str = ""
    decided_at: datetime
