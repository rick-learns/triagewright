"""Lane E: tracing, pricing, cost, TTFT, and latency on OpenTelemetry."""

from triagewright.observability.exporters import JsonlSpanExporter, span_processor, to_trace_event
from triagewright.observability.pricing import PriceTable, UnpricedModelError
from triagewright.observability.telemetry import (
    JUDGE_PURPOSE,
    OTelRequestTelemetry,
    OTelSpanHandle,
    OTelTelemetryFactory,
)

__all__ = [
    "JUDGE_PURPOSE",
    "JsonlSpanExporter",
    "OTelRequestTelemetry",
    "OTelSpanHandle",
    "OTelTelemetryFactory",
    "PriceTable",
    "UnpricedModelError",
    "span_processor",
    "to_trace_event",
]
