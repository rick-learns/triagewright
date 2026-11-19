"""Coordinator and subagent I/O.

Subagents are isolated: each receives only its typed input (never the
coordinator's transcript) and must return its typed output, validated before
the coordinator sees it. The coordinator owns the loop, turn budget, and the
final response.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field

from triagewright.contracts.common import (
    ApprovalId,
    Cents,
    Contract,
    EscalationId,
    JsonObject,
    OrderId,
    RequestId,
    TicketId,
)
from triagewright.contracts.guardrails import GuardrailVerdict
from triagewright.contracts.rag import Citation
from triagewright.contracts.telemetry import RequestMetrics
from triagewright.contracts.tickets import Ticket
from triagewright.contracts.tools import RefundEligibility, ToolName, ToolResult
from triagewright.contracts.world import EscalationQueue, EscalationReason, Priority


class SubagentName(StrEnum):
    TRIAGE = "triage"
    ANSWER = "answer"  # retrieval + grounded answer
    ACTION = "action"  # business tools
    ESCALATION = "escalation"  # escalation reviewer


class TicketCategory(StrEnum):
    ORDER_STATUS = "order_status"
    SHIPPING = "shipping"
    REFUND_REQUEST = "refund_request"
    RETURN = "return"
    BILLING = "billing"
    ACCOUNT_ACCESS = "account_access"
    PRODUCT_QUESTION = "product_question"
    POLICY_QUESTION = "policy_question"
    COMPLAINT = "complaint"
    OTHER = "other"


Confidence = Annotated[float, Field(ge=0.0, le=1.0)]


# ---- triage ---------------------------------------------------------------------


class TriageInput(Contract):
    ticket: Ticket  # body already PII-redacted


class ExtractedEntities(Contract):
    order_ids: list[OrderId] = []
    amounts_cents: list[Cents] = []
    product_names: list[str] = []


class TriageResult(Contract):
    category: TicketCategory
    intent: Annotated[str, Field(max_length=300)]
    urgency: Priority
    entities: ExtractedEntities = ExtractedEntities()
    needs_human: bool
    needs_human_reasons: list[EscalationReason] = []
    injection_suspected: bool = False
    # The customer asks for money back now, under any category; code then runs the refund gate.
    refund_requested: bool = False
    # The ticket asks support to change account records or settings; with injection
    # suspected, code sends the ticket to security.
    account_change_requested: bool = False
    missing_info: list[str] = []  # what we must ask the customer for
    confidence: Confidence


# ---- answer (retrieval + grounded response) ------------------------------------


class AnswerInput(Contract):
    ticket: Ticket
    triage: TriageResult
    facts: list[str] = []  # verified facts from tools, e.g. "Order ord_x shipped 2026-05-02"


class AnswerDraft(Contract):
    answer: str
    citations: list[Citation] = []
    grounded: bool  # every policy claim is backed by a citation
    unanswerable: bool = False
    confidence: Confidence


# ---- action (business tools) ----------------------------------------------------


class ActionInput(Contract):
    ticket: Ticket
    triage: TriageResult
    allowed_tools: list[ToolName]


class ProposedAction(Contract):
    tool: ToolName
    arguments: JsonObject  # validated against TOOL_SPECS[tool].input_model by the executor
    rationale: Annotated[str, Field(max_length=500)]


class ActionStatus(StrEnum):
    EXECUTED = "executed"
    BLOCKED = "blocked"  # guardrail said no
    PENDING_APPROVAL = "pending_approval"  # waiting on a human ApprovalDecision
    FAILED = "failed"  # tool returned is_error after retries
    SKIPPED = "skipped"


class ActionOutcome(Contract):
    action: ProposedAction
    status: ActionStatus
    result: ToolResult | None = None
    verdict: GuardrailVerdict | None = None
    approval_id: ApprovalId | None = None
    attempts: Annotated[int, Field(ge=0)] = 0
    # The executor's own eligibility lookup, for every refund that got one.
    # A blocked refund routes on ``policy.REFUND_OUTCOME[eligibility.reasons[0]]``
    # (reasons are in ``REFUND_PRECEDENCE`` order), never on verdict rule ids.
    eligibility: RefundEligibility | None = None


class ActionResult(Contract):
    outcomes: list[ActionOutcome] = []
    facts: list[str] = []  # human-readable verified facts for the answer step
    summary: str = ""


# ---- escalation reviewer --------------------------------------------------------


class EscalationInput(Contract):
    ticket: Ticket
    triage: TriageResult
    answer: AnswerDraft | None = None
    actions: ActionResult | None = None


class EscalationDecision(Contract):
    escalate: bool
    reasons: list[EscalationReason] = []
    queue: EscalationQueue | None = None
    priority: Priority = Priority.NORMAL
    handoff_summary: Annotated[str, Field(max_length=2000)] = ""  # for the human agent


# ---- coordinator ----------------------------------------------------------------


class ResolutionStatus(StrEnum):
    RESOLVED = "resolved"  # answered and/or completed with no human needed
    PENDING_APPROVAL = "pending_approval"  # irreversible action awaits a human
    ESCALATED = "escalated"  # handed to a human queue
    NEEDS_INFO = "needs_info"  # asked the customer for missing details
    REFUSED = "refused"  # out of scope or unsafe request, declined


class LoopStop(StrEnum):
    COMPLETED = "completed"
    MAX_TURNS = "max_turns"
    TIMEOUT = "timeout"
    REFUSAL = "refusal"
    ERROR = "error"


class LoopConfig(Contract):
    max_turns: Annotated[int, Field(ge=1, le=50)] = 8
    turn_timeout_s: Annotated[float, Field(gt=0)] = 20.0
    total_timeout_s: Annotated[float, Field(gt=0)] = 30.0
    handoff_reserve_s: Annotated[float, Field(ge=0)] = 3.0  # kept back for the escalation write
    max_tokens: Annotated[int, Field(ge=1)] = 1024
    tool_retries: Annotated[int, Field(ge=0, le=5)] = 2  # retries of retryable tool errors
    retry_backoff_s: Annotated[float, Field(ge=0)] = 0.5  # base; evals use 0
    # Total wait for model rate-limit slots per request. Pacing doesn't count against the
    # timeouts above; a request that would wait longer hands off instead.
    max_pacing_s: Annotated[float, Field(ge=0)] = 120.0


class FinalResponse(Contract):
    """Built by coordinator *code*. The model's ``finalize`` call supplies only ``reply``,
    citation candidates, and a proposed status among resolved/needs_info/refused;
    ``actions`` come from executor records, ``escalation`` is code's routing decision,
    and ``handoff_id`` confirms the gateway stored it. A pending approval or an
    escalation decision overrides the proposed status; if both apply, ``escalated`` wins
    over ``pending_approval``. ``escalated`` with no ``handoff_id`` means the write wasn't
    confirmed (it failed, or timed out and may still have been stored).
    """

    request_id: RequestId
    ticket_id: TicketId
    status: ResolutionStatus
    reply: str  # customer-facing, PII-safe
    citations: list[Citation] = []
    triage: TriageResult | None = None
    actions: list[ActionOutcome] = []
    escalation: EscalationDecision | None = None
    # Set only once ``create_escalation`` confirmed the stored handoff. An unconfirmed write
    # leaves the status ``escalated`` (a human must take over) with no id, and the reply
    # says so. No id isn't proof nothing was stored: a timed-out write may have committed.
    handoff_id: EscalationId | None = None
    stop: LoopStop = LoopStop.COMPLETED
    turns: Annotated[int, Field(ge=0)] = 0
    metrics: RequestMetrics | None = None
    mode: Literal["live", "replay", "stub"]
