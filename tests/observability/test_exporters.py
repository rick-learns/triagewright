from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExportResult,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from tests.observability.support import request, response
from triagewright.config import Settings
from triagewright.contracts.telemetry import SpanKind, TraceEvent
from triagewright.contracts.tools import ErrorCategory, ToolName, ToolResult
from triagewright.observability import (
    JsonlSpanExporter,
    OTelTelemetryFactory,
    exporters,
    span_processor,
    to_trace_event,
)


def run_request(factory: OTelTelemetryFactory) -> None:
    tel = factory.for_request("req_00000001", "tkt_0001")
    with tel.span("turn 1", SpanKind.AGENT_TURN), tel.span("llm", SpanKind.MODEL_CALL):
        tel.record_model_call(
            request(model="gemini-3.5-flash-lite"),
            response(model="gemini-3.5-flash-lite", ttft_ms=3),
            started_ms=0,
        )
    tel.record_tool_call(
        ToolResult.fail(ToolName.GET_ORDER, ErrorCategory.NOT_FOUND, "ord_x"), latency_ms=1
    )
    metrics = tel.finish()
    assert metrics.cost_usd == 0.0 and metrics.input_tokens == 1000


def test_jsonl_writes_one_trace_event_per_span(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, trace_exporter="jsonl", trace_dir=tmp_path / "traces")
    factory = OTelTelemetryFactory.from_settings(settings)
    run_request(factory)
    factory.shutdown()

    files = list((tmp_path / "traces").glob("*.jsonl"))
    assert len(files) == 1
    raw = [json.loads(line) for line in files[0].read_text().splitlines()]
    events = {e["name"]: TraceEvent.model_validate(e) for e in raw}
    assert set(events) == {"request", "turn 1", "llm", "tool:get_order"}
    assert files[0].name == f"{events['request'].ts.date().isoformat()}.jsonl"
    root = events["request"]
    assert root.kind is SpanKind.REQUEST and root.parent_span_id is None
    assert {e.request_id for e in events.values()} == {"req_00000001"}
    assert events["turn 1"].parent_span_id == root.span_id
    assert events["llm"].parent_span_id == events["turn 1"].span_id
    assert events["llm"].kind is SpanKind.MODEL_CALL
    assert events["llm"].attributes["tw.cost_usd"] == 0.0
    assert events["llm"].attributes["tw.ttft_ms"] == 3
    tool = events["tool:get_order"]
    assert tool.status == "error" and tool.parent_span_id == root.span_id
    assert tool.attributes["tw.tool.error_category"] == "not_found"
    for event in events.values():  # mapped to fields, not repeated as attributes
        assert "tw.kind" not in event.attributes and "tw.request_id" not in event.attributes
        assert len(event.span_id) == 16


def test_none_exporter_is_a_no_op(tmp_path: Path) -> None:
    assert span_processor("none", trace_dir=tmp_path, otlp_endpoint="http://x") is None
    settings = Settings(_env_file=None, trace_exporter="none", trace_dir=tmp_path / "traces")
    factory = OTelTelemetryFactory.from_settings(settings)
    run_request(factory)  # metrics are still computed
    factory.shutdown()
    assert not (tmp_path / "traces").exists()


def test_otlp_exports_over_http_to_the_traces_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made: list[dict[str, Any]] = []

    def fake_exporter(**kwargs: Any) -> InMemorySpanExporter:
        made.append(kwargs)
        return InMemorySpanExporter()

    monkeypatch.setattr(exporters, "OTLPSpanExporter", fake_exporter)
    processor = span_processor("otlp", trace_dir=tmp_path, otlp_endpoint="http://jaeger:4318/")
    assert isinstance(processor, BatchSpanProcessor)
    assert made == [{"endpoint": "http://jaeger:4318/v1/traces"}]
    processor.shutdown()  # type: ignore[no-untyped-call]


def test_jsonl_skips_foreign_spans_and_reports_write_failures(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    memory = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    tracer = provider.get_tracer("test")
    tracer.start_span("foreign").end()
    tracer.start_span("ours", attributes={"tw.kind": "request", "tw.request_id": "r"}).end()
    foreign, ours = memory.get_finished_spans()
    assert to_trace_event(foreign) is None

    exporter = JsonlSpanExporter(tmp_path / "out")
    assert exporter.export([foreign]) is SpanExportResult.SUCCESS
    assert list((tmp_path / "out").iterdir()) == []

    blocked = tmp_path / "file"
    blocked.write_text("")
    assert JsonlSpanExporter(blocked / "sub").export([ours]) is SpanExportResult.FAILURE
    assert "could not write traces" in caplog.text
    provider.shutdown()
