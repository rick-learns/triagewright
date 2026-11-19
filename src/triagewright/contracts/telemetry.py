"""Trace events and per-request cost/latency metrics.

Definitions used everywhere (evals, API, report):

- ``latency_ms``: wall time from request start to ``FinalResponse`` ready.
- ``ttft_ms``: time from request start to the first token of the *first* model
  call (streamed). ``None`` when no model call streamed.
- ``cost_usd``: sum of model-call costs from the pricing table. Free models
  cost 0. An unknown model is an error, never a guessed price.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field

from triagewright.contracts.common import Contract
from triagewright.contracts.llm import Provider, StopReason


class SpanKind(StrEnum):
    REQUEST = "request"
    AGENT_TURN = "agent_turn"
    SUBAGENT = "subagent"
    MODEL_CALL = "model_call"
    TOOL_CALL = "tool_call"
    RETRIEVAL = "retrieval"
    GUARDRAIL = "guardrail"
    APPROVAL = "approval"


type AttrValue = str | int | float | bool


class TraceEvent(Contract):
    """One finished span, as written to ``var/traces/*.jsonl`` and mapped to OTel."""

    ts: datetime
    request_id: str
    span_id: str
    parent_span_id: str | None = None
    name: str
    kind: SpanKind
    status: Literal["ok", "error"] = "ok"
    duration_ms: Annotated[float, Field(ge=0)]
    attributes: dict[str, AttrValue] = {}


class ModelPrice(Contract):
    model: str
    provider: Provider
    input_per_mtok_usd: Annotated[float, Field(ge=0)]
    output_per_mtok_usd: Annotated[float, Field(ge=0)]
    cache_read_per_mtok_usd: Annotated[float, Field(ge=0)] = 0.0
    cache_write_per_mtok_usd: Annotated[float, Field(ge=0)] = 0.0
    source: str  # where the price came from (URL + date checked)


class ModelCallRecord(Contract):
    purpose: str
    provider: Provider
    model: str
    input_tokens: Annotated[int, Field(ge=0)]
    output_tokens: Annotated[int, Field(ge=0)]
    cache_read_input_tokens: Annotated[int, Field(ge=0)] = 0
    cache_creation_input_tokens: Annotated[int, Field(ge=0)] = 0
    cost_usd: Annotated[float, Field(ge=0)]
    ttft_ms: Annotated[float, Field(ge=0)] | None = None
    latency_ms: Annotated[float, Field(ge=0)]
    stop_reason: StopReason


class RequestMetrics(Contract):
    request_id: str
    ticket_id: str
    model_calls: list[ModelCallRecord] = []
    tool_calls: Annotated[int, Field(ge=0)] = 0
    input_tokens: Annotated[int, Field(ge=0)] = 0
    output_tokens: Annotated[int, Field(ge=0)] = 0
    cost_usd: Annotated[float, Field(ge=0)] = 0.0
    ttft_ms: Annotated[float, Field(ge=0)] | None = None
    latency_ms: Annotated[float, Field(ge=0)] = 0.0
