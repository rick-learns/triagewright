"""Guardrail verdicts, PII redaction, refund policy, and human approval records."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated

from pydantic import Field

from triagewright.contracts.common import ApprovalId, Contract, CustomerId, JsonObject, TicketId
from triagewright.contracts.tools import RefundEligibility, ToolName


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
    max_refunds_per_customer_30d: Annotated[int, Field(ge=1)] = 2


class PolicyContext(Contract):
    """Code-owned facts for ``check_action``. Nothing here comes from model output.

    ``injection_suspected`` is request-scoped: the coordinator ORs every
    ``screen`` verdict it records (customer text, tool results) and passes it
    to every subagent and the executor. Once true, model-directed irreversible
    actions are blocked and the code-owned security handoff is used instead.
    """

    ticket_id: TicketId
    customer_id: CustomerId | None
    eligibility: RefundEligibility | None = None  # required for issue_refund, else BLOCK
    injection_suspected: bool = False
    now: datetime


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"


class ApprovalRequest(Contract):
    """A proposed irreversible action waiting for a human."""

    approval_id: ApprovalId  # common.stable_id("apr", ticket_id, tool, arguments)
    ticket_id: TicketId
    customer_id: CustomerId
    tool: ToolName
    arguments: JsonObject  # exactly what will run; never re-read from the model
    idempotency_key: str  # refund-<approval_id> for refunds
    rationale: str
    verdict: GuardrailVerdict
    status: ApprovalStatus = ApprovalStatus.PENDING
    requested_at: datetime


class ApprovalDecision(Contract):
    approval_id: ApprovalId
    approved: bool
    # A role or handle, never PII. The API/CLI authenticates the reviewer
    # (TW_APPROVER_TOKEN) before building a decision; the model never can.
    reviewer: Annotated[str, Field(min_length=1, max_length=100)]
    note: str = ""
    decided_at: datetime
