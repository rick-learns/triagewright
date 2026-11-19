"""Interfaces between lanes. Each lane implements its port; fakes live in ``triagewright.testing``.

The composition root (``triagewright.wiring``, added at integration) builds
the real objects and hands them to the coordinator.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from datetime import datetime
from typing import Protocol, runtime_checkable

from triagewright.contracts.agents import ActionOutcome, FinalResponse, ProposedAction
from triagewright.contracts.common import JsonObject
from triagewright.contracts.evals import ToolFault
from triagewright.contracts.guardrails import (
    ApprovalDecision,
    ApprovalRequest,
    ContentSource,
    GuardrailVerdict,
    PolicyContext,
    RedactionResult,
)
from triagewright.contracts.llm import ModelRequest, ModelResponse
from triagewright.contracts.rag import RetrievalQuery, RetrievedChunk
from triagewright.contracts.telemetry import AttrValue, RequestMetrics, SpanKind
from triagewright.contracts.tickets import Ticket
from triagewright.contracts.tools import (
    CallContext,
    ToolCallRecord,
    ToolName,
    ToolResult,
    ToolSpec,
)
from triagewright.contracts.world import WorldState


@runtime_checkable
class ModelClient(Protocol):
    """Lane B. One call to a model. Must not retry silently beyond its own policy."""

    async def complete(self, request: ModelRequest) -> ModelResponse: ...


@runtime_checkable
class ToolGateway(Protocol):
    """Lane A (MCP client side). Never raises for tool failures; returns ``is_error`` results.

    Enforces customer scoping from ``ctx.customer_id`` (see ``contracts.tools``).
    Does not retry: the agent's tool-call wrapper (lane B) retries retryable
    errors up to ``LoopConfig.tool_retries`` times, so ``ToolFault.times``
    counts raw gateway calls.
    """

    def list_tools(self) -> Sequence[ToolSpec]: ...

    async def call(self, name: ToolName, arguments: JsonObject, ctx: CallContext) -> ToolResult: ...


@runtime_checkable
class Retriever(Protocol):
    """Lane C."""

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]: ...


@runtime_checkable
class Guardrails(Protocol):
    """Lane D. Pure, synchronous, deterministic."""

    def redact(self, text: str) -> RedactionResult: ...

    def screen(self, text: str, source: ContentSource) -> GuardrailVerdict: ...

    def check_action(self, action: ProposedAction, ctx: PolicyContext) -> GuardrailVerdict: ...


@runtime_checkable
class ApprovalStore(Protocol):
    """Lane D. Human-in-the-loop approvals for irreversible actions."""

    def submit(self, request: ApprovalRequest) -> ApprovalRequest:
        """Insert if absent. If ``approval_id`` exists, return the stored request unchanged."""
        ...

    def get(self, approval_id: str) -> ApprovalRequest | None: ...

    def decide(self, decision: ApprovalDecision) -> ApprovalRequest: ...

    def mark_executed(self, approval_id: str) -> ApprovalRequest: ...

    def list_pending(self) -> list[ApprovalRequest]: ...


@runtime_checkable
class ActionExecutor(Protocol):
    """Lane D. The only path from a proposed write to the gateway.

    ``propose``: blocks ``tools.CODE_OWNED_TOOLS``, then validates
    ``action.arguments`` against ``tools.proposal_model(tool)`` (a
    model-supplied ``approval_id`` or token is a validation error). For an
    irreversible tool it then blocks if ``injection_suspected``, and if the
    approval for these exact arguments already exists it returns that
    approval's stored status without re-applying policy (so an executed
    refund isn't reported as refused for the spent balance). It builds
    the :class:`PolicyContext` itself from ``ctx`` and its own clock. For
    refunds it calls ``check_refund_eligibility`` (``caller="action"``,
    counted in ``ExpectedOutcome.attempts``), retrying retryable errors up to
    ``LoopConfig.tool_retries`` times; a persistent error returns ``failed``
    with that ``ToolResult``, so lane B maps it like any tool error
    (``not_found`` -> needs_info, retryable -> tool_failure). This lookup
    needs no injection screen: it is parsed into ``RefundEligibility``, which
    holds only ids, integers, dates, and ``RefundReason`` codes, so it can't
    carry instructions. Then
    ``Guardrails.check_action``: ``allow`` runs the call; ``require_approval``
    submits an :class:`ApprovalRequest` with the exact arguments and
    idempotency key and returns ``pending_approval`` (or the stored status if
    that approval already exists); ``block`` returns ``blocked``. Every
    outcome after a successful lookup carries it in ``eligibility``; lane B
    routes a blocked refund on its first reason (``policy.REFUND_OUTCOME``),
    not on ``verdict.rule_ids``. It never runs an irreversible tool.

    Lane B calls ``propose`` for irreversible actions only after the last
    ``screen`` of the request, so ``injection_suspected`` is final and no
    approval is ever stored for a request that turns out to be flagged.

    ``decide``: the one path the API and CLI use after a human decision. It
    records the decision; if approved it runs ``execute_approved``. Repeating
    an approve for an already-approved (not yet executed) request resumes
    execution with the stored idempotency key; any other repeat decision
    raises ``ValueError``.

    ``execute_approved``: loads the stored request (never fresh model
    arguments), requires status ``approved``, signs ``policy.approval_token``,
    and calls the gateway with ``caller="human"``. It does not pre-check
    eligibility: the MCP server checks, in order, token, ownership,
    idempotency key (an already-committed refund is returned as success), and
    only then eligibility for a new write. So a retry after a lost response
    succeeds instead of failing on the now-spent balance. The approval is
    marked ``executed`` only on confirmed success; after a failure it stays
    ``approved``, and calling this again retries with the same key.
    """

    async def propose(
        self, action: ProposedAction, ctx: CallContext, *, injection_suspected: bool
    ) -> ActionOutcome: ...

    async def decide(self, decision: ApprovalDecision) -> ActionOutcome: ...

    async def execute_approved(self, approval_id: str) -> ActionOutcome: ...


@runtime_checkable
class SpanHandle(Protocol):
    def set_attribute(self, key: str, value: AttrValue) -> None: ...

    def set_error(self, message: str) -> None: ...


@runtime_checkable
class RequestTelemetry(Protocol):
    """Lane E. One instance per request."""

    def span(
        self, name: str, kind: SpanKind, attributes: Mapping[str, AttrValue] | None = None
    ) -> AbstractContextManager[SpanHandle]: ...

    def record_model_call(
        self,
        request: ModelRequest,
        response: ModelResponse,
        started_ms: float,
        *,
        paced_ms: float = 0.0,
    ) -> None:
        """``started_ms``: when the call started, as an offset from the request start.
        ``paced_ms``: how much of the call waited for rate-limit slots."""

    def record_pacing(self, paced_ms: float) -> None:
        """Rate-limit waits this request made, failed calls included. Excluded from latency."""

    def record_tool_call(self, result: ToolResult, latency_ms: float) -> None: ...

    def finish(self) -> RequestMetrics: ...


@runtime_checkable
class TelemetryFactory(Protocol):
    def for_request(self, request_id: str, ticket_id: str) -> RequestTelemetry: ...


@runtime_checkable
class TicketResolver(Protocol):
    """Lane B (coordinator). The single entry point the API, CLI, and evals call."""

    async def resolve(self, ticket: Ticket) -> FinalResponse: ...


@runtime_checkable
class CaseRuntime(Protocol):
    """One isolated stack for one eval case: its own world, clock, faults, and logs."""

    @property
    def resolver(self) -> TicketResolver: ...

    def tool_calls(self) -> list[ToolCallRecord]:
        """Every raw gateway call so far, in order (including retries and faults)."""
        ...

    def ledger(self) -> WorldState:
        """Current state of the mock systems (refunds, escalations, ...)."""
        ...

    def pending_approvals(self) -> list[ApprovalRequest]: ...


@runtime_checkable
class CaseRuntimeFactory(Protocol):
    """Builds a fresh :class:`CaseRuntime` per case. Eval knobs never touch the production API.

    The real factory (``triagewright.wiring``, integration) runs the MCP server
    in-process on a fresh SQLite database. ``triagewright.testing`` has a fake.
    """

    def build(
        self, *, case_id: str, world: WorldState, as_of: datetime, faults: Sequence[ToolFault]
    ) -> CaseRuntime:
        """``case_id`` selects the case's recordings (see ``llm.Recording``)."""
        ...
