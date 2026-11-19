"""OpenTelemetry implementation of the ``TelemetryFactory`` / ``RequestTelemetry`` ports.

Each request is a root span; :meth:`OTelRequestTelemetry.span` nests spans
under the caller's current span (tracked per asyncio task, so parallel
subagents keep their own parents). ``record_model_call`` and
``record_tool_call`` put their attributes on the enclosing span when it is an
unrecorded span of the matching kind, and otherwise emit their own child span.

Metrics follow ``contracts.telemetry``. In replay, ``latency_ms`` is
estimated: measured wall time with the measured model time swapped for each
call's recorded latency. Eval-judge calls (purpose ``"judge"``) are traced but
never counted in the request's metrics; they are kept in ``judge_calls`` for
the eval report.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.trace import Status, StatusCode

from triagewright.config import Settings
from triagewright.contracts.llm import ModelRequest, ModelResponse, Provider
from triagewright.contracts.telemetry import AttrValue, ModelCallRecord, RequestMetrics, SpanKind
from triagewright.contracts.tools import ToolResult
from triagewright.observability.exporters import KIND_ATTR, REQUEST_ID_ATTR, span_processor
from triagewright.observability.pricing import PriceTable, UnpricedModelError

JUDGE_PURPOSE = "judge"


@dataclass
class _Node:
    owner: OTelRequestTelemetry
    span: trace.Span
    kind: SpanKind
    recorded: bool = False


_current: ContextVar[_Node | None] = ContextVar("triagewright_current_span", default=None)


class OTelSpanHandle:
    def __init__(self, span: trace.Span) -> None:
        self._span = span

    def set_attribute(self, key: str, value: AttrValue) -> None:
        self._span.set_attribute(key, value)

    def set_error(self, message: str) -> None:
        self._span.set_status(Status(StatusCode.ERROR, message))


class OTelRequestTelemetry:
    """One request's spans and metrics. Times are offsets in ms from construction."""

    def __init__(
        self,
        tracer: trace.Tracer,
        prices: PriceTable,
        request_id: str,
        ticket_id: str,
        *,
        replay: bool,
        clock: Callable[[], float],
    ) -> None:
        self.request_id = request_id
        self.ticket_id = ticket_id
        self.judge_calls: list[ModelCallRecord] = []
        self._tracer = tracer
        self._prices = prices
        self._replay = replay
        self._clock = clock
        self._t0 = clock()
        self._t0_ns = time.time_ns()
        self._calls: list[ModelCallRecord] = []
        # (measured start, measured end, recorded latency) per model call, for replay.
        self._model_times: list[tuple[float, float, float]] = []
        self._tool_calls = 0
        self._metrics: RequestMetrics | None = None
        root = tracer.start_span(
            "request",
            context=Context(),  # always a root, whatever span is ambient
            start_time=self._t0_ns,
            attributes={**self._base(SpanKind.REQUEST), "tw.ticket_id": ticket_id},
        )
        self._root = _Node(self, root, SpanKind.REQUEST)

    def _now_ms(self) -> float:
        return (self._clock() - self._t0) * 1000

    def _ns(self, offset_ms: float) -> int:
        return self._t0_ns + int(offset_ms * 1_000_000)

    def _base(self, kind: SpanKind) -> dict[str, AttrValue]:
        return {KIND_ATTR: kind.value, REQUEST_ID_ATTR: self.request_id}

    def _parent(self) -> _Node:
        node = _current.get()
        return node if node is not None and node.owner is self else self._root

    @contextmanager
    def span(
        self, name: str, kind: SpanKind, attributes: Mapping[str, AttrValue] | None = None
    ) -> Iterator[OTelSpanHandle]:
        otel = self._tracer.start_span(
            name,
            context=trace.set_span_in_context(self._parent().span),
            start_time=self._ns(self._now_ms()),
            attributes={**(attributes or {}), **self._base(kind)},
        )
        token = _current.set(_Node(self, otel, kind))
        try:
            yield OTelSpanHandle(otel)
        except BaseException as exc:
            otel.set_status(Status(StatusCode.ERROR, f"{type(exc).__name__}: {exc}"))
            raise
        finally:
            _current.reset(token)
            otel.end(end_time=self._ns(self._now_ms()))

    def _annotate(
        self,
        kind: SpanKind,
        name: str,
        start_ms: float,
        end_ms: float,
        attributes: dict[str, AttrValue],
        error: str | None = None,
    ) -> None:
        """Set ``attributes`` on the enclosing span of ``kind``, or emit a child span."""
        parent = self._parent()
        if parent.kind is kind and not parent.recorded:
            parent.recorded = True
            span = parent.span
            span.set_attributes(attributes)
        else:
            span = self._tracer.start_span(
                name,
                context=trace.set_span_in_context(parent.span),
                start_time=self._ns(start_ms),
                attributes={**attributes, **self._base(kind)},
            )
        if error is not None:
            span.set_status(Status(StatusCode.ERROR, error))
        if span is not parent.span:
            span.end(end_time=self._ns(end_ms))

    def _replay_latency_ms(self, end_ms: float) -> float:
        """Wall time minus the measured model time, plus each call's recorded latency."""
        measured = 0.0
        reach = 0.0
        for start, end, _ in sorted(self._model_times):
            start = max(start, reach)
            if end > start:
                measured += end - start
                reach = end
        recorded = sum(rec for _, _, rec in self._model_times)
        return max(end_ms - measured + recorded, 0.0)

    def record_model_call(
        self, request: ModelRequest, response: ModelResponse, started_ms: float
    ) -> None:
        end_ms = self._now_ms()
        start_ms = min(started_ms, end_ms)
        cost = self._prices.response_cost(response)
        if cost is None:
            raise UnpricedModelError(
                f"no price for {response.provider}/{response.model} and no provider cost"
            )
        usage = response.usage
        record = ModelCallRecord(
            purpose=request.purpose,
            started_ms=started_ms,
            provider=response.provider,
            model=response.model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_input_tokens=usage.cache_read_input_tokens,
            cache_creation_input_tokens=usage.cache_creation_input_tokens,
            cost_usd=cost,
            ttft_ms=response.ttft_ms,
            latency_ms=response.latency_ms,
            stop_reason=response.stop_reason,
            paced_ms=response.paced_ms,
        )
        judge = request.purpose == JUDGE_PURPOSE
        # A judge call's measured time still leaves the wall time; its recording never joins it.
        self._model_times.append((start_ms, end_ms, 0.0 if judge else response.latency_ms))
        (self.judge_calls if judge else self._calls).append(record)
        attrs: dict[str, AttrValue] = {
            "tw.purpose": request.purpose,
            "gen_ai.system": response.provider.value,
            "gen_ai.request.model": request.model,
            "gen_ai.response.model": response.model,
            "gen_ai.usage.input_tokens": usage.input_tokens,
            "gen_ai.usage.output_tokens": usage.output_tokens,
            "tw.usage.cache_read_input_tokens": usage.cache_read_input_tokens,
            "tw.usage.cache_creation_input_tokens": usage.cache_creation_input_tokens,
            "tw.cost_usd": cost,
            "tw.latency_ms": response.latency_ms,
            "tw.stop_reason": response.stop_reason.value,
            "tw.estimated": self._replay,
            "tw.judge": judge,
        }
        if response.ttft_ms is not None:
            attrs["tw.ttft_ms"] = response.ttft_ms
        name = f"model:{request.purpose}"
        self._annotate(SpanKind.MODEL_CALL, name, start_ms, end_ms, attrs)

    def record_tool_call(self, result: ToolResult, latency_ms: float) -> None:
        self._tool_calls += 1
        end_ms = self._now_ms()
        attrs: dict[str, AttrValue] = {
            "tw.tool": result.tool.value,
            "tw.tool.is_error": result.is_error,
            "tw.latency_ms": latency_ms,
        }
        error = None
        if result.error is not None:
            error = result.error.category.value
            attrs["tw.tool.error_category"] = error
            attrs["tw.tool.retryable"] = result.error.retryable
        name = f"tool:{result.tool.value}"
        start_ms = max(end_ms - latency_ms, 0.0)
        self._annotate(SpanKind.TOOL_CALL, name, start_ms, end_ms, attrs, error)

    def finish(self) -> RequestMetrics:
        """Close the request span and return its metrics. Later calls return the same metrics."""
        if self._metrics is not None:
            return self._metrics
        end_ms = self._now_ms()
        calls = self._calls
        first = min(calls, key=lambda c: c.started_ms, default=None)
        ttft = None
        if first is not None and first.ttft_ms is not None:
            ttft = first.started_ms + first.paced_ms + first.ttft_ms
        paced = sum(c.paced_ms for c in calls)
        metrics = RequestMetrics(
            request_id=self.request_id,
            ticket_id=self.ticket_id,
            model_calls=list(calls),
            tool_calls=self._tool_calls,
            input_tokens=sum(c.input_tokens for c in calls),
            output_tokens=sum(c.output_tokens for c in calls),
            cost_usd=sum(c.cost_usd for c in calls),
            ttft_ms=ttft,
            latency_ms=self._replay_latency_ms(end_ms) if self._replay else max(end_ms - paced, 0),
            paced_ms=paced,
            estimated=self._replay,
        )
        totals: dict[str, AttrValue] = {
            "tw.model_calls": len(calls),
            "tw.tool_calls": metrics.tool_calls,
            "gen_ai.usage.input_tokens": metrics.input_tokens,
            "gen_ai.usage.output_tokens": metrics.output_tokens,
            "tw.cost_usd": metrics.cost_usd,
            "tw.latency_ms": metrics.latency_ms,
            "tw.paced_ms": metrics.paced_ms,
            "tw.estimated": metrics.estimated,
        }
        if ttft is not None:
            totals["tw.ttft_ms"] = ttft
        self._root.span.set_attributes(totals)
        self._root.span.end(end_time=self._ns(end_ms))
        self._metrics = metrics
        return metrics


class OTelTelemetryFactory:
    """Builds one :class:`OTelRequestTelemetry` per request on a private tracer provider.

    ``processor=None`` exports nothing (a no-op tracer); metrics are still computed.
    """

    def __init__(
        self,
        prices: PriceTable,
        *,
        processor: SpanProcessor | None = None,
        replay: bool = False,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.prices = prices
        self._replay = replay
        self._clock = clock
        self._provider: TracerProvider | None = None
        self._tracer: trace.Tracer = trace.NoOpTracer()
        if processor is not None:
            self._provider = TracerProvider(
                resource=Resource.create({"service.name": "triagewright"})
            )
            self._provider.add_span_processor(processor)
            self._tracer = self._provider.get_tracer("triagewright")

    @classmethod
    def from_settings(cls, settings: Settings) -> OTelTelemetryFactory:
        """Load prices, fail fast on any unpriced configured model, and pick the exporter."""
        prices = PriceTable.load(settings.pricing_file)
        prices.check(settings.llm_provider, settings.models())
        processor = span_processor(
            settings.trace_exporter,
            trace_dir=settings.trace_dir,
            otlp_endpoint=settings.otlp_endpoint,
        )
        return cls(prices, processor=processor, replay=settings.llm_provider is Provider.REPLAY)

    def for_request(self, request_id: str, ticket_id: str) -> OTelRequestTelemetry:
        return OTelRequestTelemetry(
            self._tracer,
            self.prices,
            request_id,
            ticket_id,
            replay=self._replay,
            clock=self._clock,
        )

    def shutdown(self) -> None:
        """Flush and stop the exporter (call on process exit)."""
        if self._provider is not None:
            self._provider.shutdown()
