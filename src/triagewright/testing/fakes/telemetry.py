"""Null telemetry that counts calls and tokens and records spans for assertions."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager

from triagewright.contracts.llm import ModelRequest, ModelResponse
from triagewright.contracts.telemetry import AttrValue, RequestMetrics, SpanKind
from triagewright.contracts.tools import (
    ToolResult,
)


class _NullSpan:
    def __init__(self) -> None:
        self.attributes: dict[str, AttrValue] = {}
        self.error: str | None = None

    def set_attribute(self, key: str, value: AttrValue) -> None:
        self.attributes[key] = value

    def set_error(self, message: str) -> None:
        self.error = message


class NullTelemetry:
    """Counts calls and tokens; costs are always 0. Records span names for assertions."""

    def __init__(self, request_id: str = "req_00000000", ticket_id: str = "tkt_0000") -> None:
        self.request_id = request_id
        self.ticket_id = ticket_id
        self.spans: list[tuple[str, SpanKind]] = []
        self.span_errors: list[tuple[str, str]] = []
        self.responses: list[ModelResponse] = []
        self.tool_calls = 0
        self.paced_ms = 0.0

    @contextmanager
    def span(
        self, name: str, kind: SpanKind, attributes: Mapping[str, AttrValue] | None = None
    ) -> Iterator[_NullSpan]:
        self.spans.append((name, kind))
        span = _NullSpan()
        try:
            yield span
        finally:
            if span.error is not None:
                self.span_errors.append((name, span.error))

    def record_model_call(
        self,
        request: ModelRequest,
        response: ModelResponse,
        started_ms: float,
        *,
        paced_ms: float = 0.0,
    ) -> None:
        self.responses.append(response)

    def record_pacing(self, paced_ms: float) -> None:
        self.paced_ms += paced_ms

    def record_tool_call(self, result: ToolResult, latency_ms: float) -> None:
        self.tool_calls += 1

    def finish(self) -> RequestMetrics:
        return RequestMetrics(
            request_id=self.request_id,
            ticket_id=self.ticket_id,
            tool_calls=self.tool_calls,
            input_tokens=sum(r.usage.input_tokens for r in self.responses),
            output_tokens=sum(r.usage.output_tokens for r in self.responses),
        )


class NullTelemetryFactory:
    def for_request(self, request_id: str, ticket_id: str) -> NullTelemetry:
        return NullTelemetry(request_id, ticket_id)
