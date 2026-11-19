from __future__ import annotations

import asyncio

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.trace import StatusCode

from tests.observability.support import PAID, FakeClock, prices, request, response
from triagewright.contracts.llm import Provider, StopReason, Usage
from triagewright.contracts.ports import RequestTelemetry, SpanHandle, TelemetryFactory
from triagewright.contracts.telemetry import SpanKind
from triagewright.contracts.tools import ErrorCategory, ToolName, ToolResult
from triagewright.observability import (
    JUDGE_PURPOSE,
    OTelRequestTelemetry,
    OTelSpanHandle,
    OTelTelemetryFactory,
    UnpricedModelError,
)

OK = ToolResult(tool=ToolName.GET_ORDER, is_error=False, data={"order_id": "ord_a1001"})
TIMEOUT = ToolResult.fail(ToolName.GET_ORDER, ErrorCategory.TIMEOUT, "slow")


def telemetry(
    clock: FakeClock, *, replay: bool = False, exporter: InMemorySpanExporter | None = None
) -> OTelRequestTelemetry:
    processor = SimpleSpanProcessor(exporter) if exporter else None
    factory = OTelTelemetryFactory(prices(), processor=processor, replay=replay, clock=clock)
    return factory.for_request("req_00000001", "tkt_0001")


def by_name(exporter: InMemorySpanExporter) -> dict[str, ReadableSpan]:
    return {s.name: s for s in exporter.get_finished_spans()}


def test_implements_the_ports() -> None:
    factory = OTelTelemetryFactory(prices())
    assert isinstance(factory, TelemetryFactory)
    tel = factory.for_request("req_00000001", "tkt_0001")
    assert isinstance(tel, RequestTelemetry)
    with tel.span("turn", SpanKind.AGENT_TURN) as span:
        assert isinstance(span, SpanHandle) and isinstance(span, OTelSpanHandle)


def test_live_metrics_use_wall_time_and_token_totals() -> None:
    clock = FakeClock()
    tel = telemetry(clock)
    clock.advance(10)
    clock.advance(300)
    first = response(
        latency_ms=290,
        ttft_ms=50,
        usage=Usage(input_tokens=1000, output_tokens=200, cache_read_input_tokens=4000),
        stop_reason=StopReason.TOOL_USE,
    )
    tel.record_model_call(request("triage"), first, started_ms=10)
    clock.advance(20)
    tel.record_tool_call(OK, latency_ms=20)
    clock.advance(100)
    tel.record_model_call(request("answer"), response(latency_ms=100), started_ms=330)
    clock.advance(5)
    m = tel.finish()
    assert m.request_id == "req_00000001" and m.ticket_id == "tkt_0001"
    assert m.estimated is False
    assert m.latency_ms == pytest.approx(435)
    assert m.ttft_ms == pytest.approx(60)  # started_ms + that call's ttft
    assert m.tool_calls == 1
    assert (m.input_tokens, m.output_tokens) == (2000, 400)  # input excludes cache reads
    # (1000 * 1.0 + 200 * 4.0 + 4000 * 0.25) / 1e6 + (1000 * 1.0 + 200 * 4.0) / 1e6
    assert m.cost_usd == pytest.approx(0.0028 + 0.0018)
    assert [c.purpose for c in m.model_calls] == ["triage", "answer"]
    triage = m.model_calls[0]
    assert (triage.started_ms, triage.latency_ms, triage.ttft_ms) == (10, 290, 50)
    assert triage.cache_read_input_tokens == 4000
    assert triage.stop_reason is StopReason.TOOL_USE
    assert triage.provider is Provider.GEMINI and triage.model == PAID
    assert m.cost_usd == pytest.approx(sum(c.cost_usd for c in m.model_calls))


def test_pacing_waits_are_reported_apart_from_live_latency() -> None:
    clock = FakeClock()
    tel = telemetry(clock)
    clock.advance(4000 + 300)  # 4 s waiting for a rate-limit slot, then the call
    tel.record_pacing(4000)
    paced = response(latency_ms=300, ttft_ms=50)
    tel.record_model_call(request("triage"), paced, started_ms=0, paced_ms=4000)
    clock.advance(10)
    m = tel.finish()
    assert m.latency_ms == pytest.approx(310) and m.paced_ms == 4000
    assert m.ttft_ms == pytest.approx(4050)  # the first token still came 4.05 s in
    assert m.model_calls[0].paced_ms == 4000


def test_only_measured_pacing_leaves_latency() -> None:
    # A failed call's wait still counts; a response never says how long it was paced.
    clock = FakeClock()
    tel = telemetry(clock)
    clock.advance(80)
    tel.record_pacing(80)  # a paced call that then failed
    clock.advance(100)
    tel.record_model_call(request("triage"), response(latency_ms=100), started_ms=80)
    m = tel.finish()
    assert m.paced_ms == 80 and m.latency_ms == pytest.approx(100)


def test_replay_swaps_measured_model_time_for_recorded_latency() -> None:
    clock = FakeClock()
    tel = telemetry(clock, replay=True)
    clock.advance(10)  # non-model work
    clock.advance(1)  # replayed call returns almost instantly
    tel.record_model_call(request(), response(latency_ms=400, ttft_ms=80), started_ms=10)
    clock.advance(20)  # tool work
    tel.record_tool_call(OK, latency_ms=20)
    clock.advance(1)
    tel.record_model_call(request("answer"), response(latency_ms=600), started_ms=31)
    clock.advance(5)
    m = tel.finish()
    assert m.estimated is True
    # wall 37 ms, minus 2 ms measured model time, plus 400 + 600 recorded.
    assert m.latency_ms == pytest.approx(1035)
    assert m.ttft_ms == pytest.approx(90)
    assert [c.started_ms for c in m.model_calls] == [10, 31]  # as the caller measured them
    assert [c.latency_ms for c in m.model_calls] == [400, 600]


def test_replay_counts_overlapping_model_time_once() -> None:
    clock = FakeClock()
    tel = telemetry(clock, replay=True)
    clock.advance(10)
    # Two parallel calls both measured over [0, 10].
    tel.record_model_call(request("a"), response(latency_ms=100), started_ms=0)
    tel.record_model_call(request("b"), response(latency_ms=200), started_ms=0)
    m = tel.finish()
    assert m.latency_ms == pytest.approx(10 - 10 + 300)


def test_replay_estimate_ignores_judge_recordings() -> None:
    clock = FakeClock()
    tel = telemetry(clock, replay=True)
    clock.advance(4)
    tel.record_model_call(request("answer"), response(latency_ms=100), started_ms=2)
    clock.advance(2)
    tel.record_model_call(request(JUDGE_PURPOSE), response(latency_ms=900), started_ms=4)
    # wall 6 ms, minus 4 ms measured model time (answer and judge), plus 100 recorded.
    assert tel.finish().latency_ms == pytest.approx(102)


def test_ttft_is_none_when_the_first_call_did_not_stream() -> None:
    clock = FakeClock()
    tel = telemetry(clock)
    assert tel.finish().ttft_ms is None
    tel = telemetry(clock)
    tel.record_model_call(request("triage"), response(), started_ms=0)
    tel.record_model_call(request("answer"), response(ttft_ms=5), started_ms=0.5)
    assert tel.finish().ttft_ms is None


def test_judge_calls_are_excluded_from_ticket_metrics() -> None:
    clock = FakeClock()
    tel = telemetry(clock)
    tel.record_model_call(request("answer"), response(ttft_ms=7), started_ms=0)
    tel.record_model_call(
        request(JUDGE_PURPOSE), response(ttft_ms=1, provider_cost_usd=0.5), started_ms=0
    )
    m = tel.finish()
    assert [c.purpose for c in m.model_calls] == ["answer"]
    assert m.cost_usd == pytest.approx(0.0018) and m.input_tokens == 1000
    assert m.ttft_ms == 7
    assert [c.cost_usd for c in tel.judge_calls] == [0.5]


def test_provider_cost_wins_and_unknown_model_is_an_error() -> None:
    clock = FakeClock()
    tel = telemetry(clock)
    tel.record_model_call(request(), response(provider_cost_usd=0.0), started_ms=0)
    tel.record_model_call(
        request(), response(model="unlisted", provider_cost_usd=0.01), started_ms=0
    )
    with pytest.raises(UnpricedModelError, match="unlisted"):
        tel.record_model_call(request(), response(model="unlisted"), started_ms=0)
    m = tel.finish()
    assert [c.cost_usd for c in m.model_calls] == [0.0, 0.01]


def test_finish_is_idempotent() -> None:
    clock = FakeClock()
    exporter = InMemorySpanExporter()
    tel = telemetry(clock, exporter=exporter)
    first = tel.finish()
    clock.advance(50)
    assert tel.finish() is first
    assert len(exporter.get_finished_spans()) == 1


def test_spans_nest_and_carry_call_attributes() -> None:
    clock = FakeClock()
    exporter = InMemorySpanExporter()
    tel = telemetry(clock, exporter=exporter)
    with tel.span("refund", SpanKind.SUBAGENT, {"tw.agent": "action"}) as sub:
        sub.set_attribute("tw.turn", 1)
        with tel.span("llm", SpanKind.MODEL_CALL):
            clock.advance(30)
            tel.record_model_call(
                request("action"), response(latency_ms=30, ttft_ms=12), started_ms=0
            )
        clock.advance(5)
        tel.record_tool_call(TIMEOUT, latency_ms=5)  # no tool span open: emits its own
        with tel.span("get_order", SpanKind.TOOL_CALL):
            tel.record_tool_call(OK, latency_ms=0)
    tel.finish()

    spans = by_name(exporter)
    assert set(spans) == {"request", "refund", "llm", "tool:get_order", "get_order"}
    root, refund = spans["request"], spans["refund"]
    assert root.parent is None and root.context is not None and refund.context is not None
    assert refund.parent is not None and refund.parent.span_id == root.context.span_id
    for child in ("llm", "tool:get_order", "get_order"):
        parent = spans[child].parent
        assert parent is not None and parent.span_id == refund.context.span_id
    assert refund.attributes is not None
    assert refund.attributes["tw.agent"] == "action" and refund.attributes["tw.turn"] == 1
    assert refund.attributes["tw.kind"] == "subagent"

    llm = spans["llm"].attributes
    assert llm is not None
    assert llm["gen_ai.usage.input_tokens"] == 1000 and llm["gen_ai.usage.output_tokens"] == 200
    assert llm["tw.cost_usd"] == pytest.approx(0.0018)
    assert (llm["tw.ttft_ms"], llm["tw.latency_ms"]) == (12, 30)
    assert llm["tw.stop_reason"] == "end_turn" and llm["tw.purpose"] == "action"

    failed = spans["tool:get_order"]
    assert failed.status.status_code is StatusCode.ERROR
    assert failed.attributes is not None
    assert failed.attributes["tw.tool.error_category"] == "timeout"
    assert failed.attributes["tw.tool.retryable"] is True
    assert failed.attributes["tw.kind"] == "tool_call"
    assert spans["get_order"].status.status_code is StatusCode.UNSET
    assert spans["get_order"].attributes is not None
    assert spans["get_order"].attributes["tw.tool.is_error"] is False

    assert root.attributes is not None
    assert root.attributes["tw.tool_calls"] == 2 and root.attributes["tw.ttft_ms"] == 12
    assert root.attributes["tw.ticket_id"] == "tkt_0001"


def test_a_second_record_in_one_model_span_gets_its_own_span() -> None:
    exporter = InMemorySpanExporter()
    tel = telemetry(FakeClock(), exporter=exporter)
    with tel.span("llm", SpanKind.MODEL_CALL):
        tel.record_model_call(request("a"), response(), started_ms=0)
        tel.record_model_call(request("b"), response(), started_ms=0)
    tel.finish()
    assert {s.name for s in exporter.get_finished_spans()} == {"request", "llm", "model:b"}


def test_span_errors_are_recorded() -> None:
    exporter = InMemorySpanExporter()
    tel = telemetry(FakeClock(), exporter=exporter)
    with tel.span("guard", SpanKind.GUARDRAIL) as span:
        span.set_error("blocked")
    with pytest.raises(RuntimeError), tel.span("boom", SpanKind.RETRIEVAL):
        raise RuntimeError("index missing for the customer's question")

    def fail_after_setting_an_error() -> None:
        with tel.span("model", SpanKind.MODEL_CALL) as span:
            span.set_error("ModelUnavailableError (HTTP 503)")
            raise RuntimeError("model call failed")

    with pytest.raises(RuntimeError):
        fail_after_setting_an_error()
    tel.finish()
    spans = by_name(exporter)
    assert spans["guard"].status.description == "blocked"
    assert spans["boom"].status.status_code is StatusCode.ERROR
    assert spans["boom"].status.description == "RuntimeError"  # never the message
    assert spans["model"].status.description == "ModelUnavailableError (HTTP 503)"


@pytest.mark.anyio
async def test_parallel_subagents_keep_their_own_parents() -> None:
    exporter = InMemorySpanExporter()
    tel = telemetry(FakeClock(), exporter=exporter)

    async def subagent(name: str) -> None:
        with tel.span(name, SpanKind.SUBAGENT):
            await asyncio.sleep(0)
            tel.record_model_call(request(name), response(), started_ms=0)

    await asyncio.gather(subagent("triage"), subagent("answer"))
    tel.finish()
    spans = by_name(exporter)
    for name in ("triage", "answer"):
        ctx, parent = spans[name].context, spans[f"model:{name}"].parent
        assert ctx is not None and parent is not None and parent.span_id == ctx.span_id


def test_request_span_is_a_root_under_an_ambient_span() -> None:
    exporter = InMemorySpanExporter()
    ambient = TracerProvider(sampler=ALWAYS_OFF).get_tracer("app")
    with ambient.start_as_current_span("unsampled"):
        tel = telemetry(FakeClock(), exporter=exporter)
    tel.finish()
    (root,) = exporter.get_finished_spans()
    assert root.name == "request" and root.parent is None


def test_spans_of_another_request_do_not_become_parents() -> None:
    exporter = InMemorySpanExporter()
    factory = OTelTelemetryFactory(prices(), processor=SimpleSpanProcessor(exporter))
    a = factory.for_request("req_0000000a", "tkt_000a")
    b = factory.for_request("req_0000000b", "tkt_000b")
    with a.span("turn", SpanKind.AGENT_TURN):
        b.record_tool_call(OK, latency_ms=0)
    b.finish()
    a.finish()
    spans = exporter.get_finished_spans()
    tool = next(s for s in spans if s.name == "tool:get_order")
    root_b = next(
        s
        for s in spans
        if s.name == "request" and s.attributes and s.attributes["tw.request_id"] == "req_0000000b"
    )
    assert tool.parent is not None and root_b.context is not None
    assert tool.parent.span_id == root_b.context.span_id
    factory.shutdown()


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
