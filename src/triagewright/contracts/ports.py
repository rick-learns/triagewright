"""Interfaces between lanes. Each lane implements its port; fakes live in ``triagewright.testing``.

The composition root (``triagewright.wiring``, added at integration) builds
the real objects and hands them to the coordinator.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from typing import Protocol, runtime_checkable

from triagewright.contracts.agents import FinalResponse, ProposedAction
from triagewright.contracts.common import JsonObject
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
from triagewright.contracts.tools import CallContext, ToolName, ToolResult, ToolSpec


@runtime_checkable
class ModelClient(Protocol):
    """Lane B. One call to a model. Must not retry silently beyond its own policy."""

    async def complete(self, request: ModelRequest) -> ModelResponse: ...


@runtime_checkable
class ToolGateway(Protocol):
    """Lane A (MCP client side). Never raises for tool failures; returns ``is_error`` results."""

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

    def submit(self, request: ApprovalRequest) -> ApprovalRequest: ...

    def get(self, approval_id: str) -> ApprovalRequest | None: ...

    def decide(self, decision: ApprovalDecision) -> ApprovalRequest: ...

    def mark_executed(self, approval_id: str) -> ApprovalRequest: ...

    def list_pending(self) -> list[ApprovalRequest]: ...


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

    def record_model_call(self, request: ModelRequest, response: ModelResponse) -> None: ...

    def record_tool_call(self, result: ToolResult, latency_ms: float) -> None: ...

    def finish(self) -> RequestMetrics: ...


@runtime_checkable
class TelemetryFactory(Protocol):
    def for_request(self, request_id: str, ticket_id: str) -> RequestTelemetry: ...


@runtime_checkable
class TicketResolver(Protocol):
    """Lane B (coordinator). The single entry point the API, CLI, and evals call."""

    async def resolve(self, ticket: Ticket) -> FinalResponse: ...
