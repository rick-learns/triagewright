"""Span exporters selected by ``Settings.trace_exporter``: ``none``, ``jsonl``, or ``otlp``.

JSONL writes one :class:`TraceEvent` per finished span to
``<trace_dir>/<UTC date>.jsonl``. OTLP sends spans over HTTP (Jaeger in the
Compose ``tracing`` profile). There is no console exporter.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import StatusCode

from triagewright.contracts.telemetry import AttrValue, SpanKind, TraceEvent

# Span attributes that carry TraceEvent fields rather than free-form attributes.
KIND_ATTR = "tw.kind"
REQUEST_ID_ATTR = "tw.request_id"


def to_trace_event(span: ReadableSpan) -> TraceEvent | None:
    """Map a finished span to a :class:`TraceEvent`; ``None`` for spans we didn't create."""
    attrs = dict(span.attributes or {})
    kind, request_id = attrs.pop(KIND_ATTR, None), attrs.pop(REQUEST_ID_ATTR, None)
    if not isinstance(kind, str) or not isinstance(request_id, str) or span.context is None:
        return None
    start, end = span.start_time or 0, span.end_time or span.start_time or 0
    scalars: dict[str, AttrValue] = {
        k: v for k, v in attrs.items() if isinstance(v, str | int | float | bool)
    }
    error = span.status.status_code is StatusCode.ERROR
    return TraceEvent(
        ts=datetime.fromtimestamp(start / 1e9, UTC),
        request_id=request_id,
        span_id=format(span.context.span_id, "016x"),
        parent_span_id=format(span.parent.span_id, "016x") if span.parent else None,
        name=span.name,
        kind=SpanKind(kind),
        status="error" if error else "ok",
        duration_ms=max(end - start, 0) / 1e6,
        attributes=scalars,
    )


class JsonlSpanExporter(SpanExporter):
    def __init__(self, trace_dir: Path) -> None:
        self.trace_dir = trace_dir
        self._lock = threading.Lock()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        by_file: dict[Path, list[str]] = {}
        for span in spans:
            if (event := to_trace_event(span)) is not None:
                path = self.trace_dir / f"{event.ts.date().isoformat()}.jsonl"
                by_file.setdefault(path, []).append(event.model_dump_json() + "\n")
        with self._lock:
            try:
                self.trace_dir.mkdir(parents=True, exist_ok=True)
                for path, lines in by_file.items():
                    with path.open("a", encoding="utf-8") as f:
                        f.writelines(lines)
            except OSError:
                return SpanExportResult.FAILURE
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


def span_processor(
    exporter: Literal["none", "jsonl", "otlp"], *, trace_dir: Path, otlp_endpoint: str
) -> SpanProcessor | None:
    """The processor for ``exporter``; ``None`` means spans are not exported at all."""
    match exporter:
        case "none":
            return None
        case "jsonl":
            return SimpleSpanProcessor(JsonlSpanExporter(trace_dir))
        case "otlp":
            endpoint = otlp_endpoint.rstrip("/") + "/v1/traces"
            return BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
