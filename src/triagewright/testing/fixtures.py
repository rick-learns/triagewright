"""Builders for common test objects."""

from __future__ import annotations

import itertools
from datetime import datetime
from functools import cache
from pathlib import Path

from triagewright.contracts.common import JsonObject
from triagewright.contracts.llm import (
    ContentBlock,
    ModelResponse,
    Provider,
    StopReason,
    TextBlock,
    ToolUseBlock,
    Usage,
)
from triagewright.contracts.rag import KBDocument
from triagewright.contracts.tickets import Ticket
from triagewright.contracts.world import WorldState
from triagewright.data import load_as_of, load_kb, load_world

REPO_ROOT = Path(__file__).resolve().parents[3]
_ids = itertools.count(1)


@cache
def seed_world() -> WorldState:
    return load_world(REPO_ROOT / "data" / "seed")


@cache
def seed_as_of() -> datetime:
    return load_as_of(REPO_ROOT / "data" / "seed")


@cache
def _kb() -> tuple[KBDocument, ...]:
    return tuple(load_kb(REPO_ROOT / "data" / "kb"))


def seed_kb() -> list[KBDocument]:
    return list(_kb())


def sample_ticket(
    body: str = "Where is my order ord_a1002?",
    *,
    ticket_id: str = "tkt_0001",
    customer_id: str | None = "cus_ava01",
    subject: str = "Order question",
) -> Ticket:
    return Ticket(
        ticket_id=ticket_id,
        customer_id=customer_id,
        subject=subject,
        body=body,
        created_at=seed_as_of(),
    )


def text_block(text: str) -> TextBlock:
    return TextBlock(text=text)


def tool_use_block(name: str, tool_input: JsonObject, block_id: str | None = None) -> ToolUseBlock:
    return ToolUseBlock(id=block_id or f"toolu_{next(_ids):04d}", name=name, input=tool_input)


def model_response(
    *content: ContentBlock,
    stop_reason: StopReason | None = None,
    model: str = "test-model",
    input_tokens: int = 100,
    output_tokens: int = 20,
) -> ModelResponse:
    """Build a response; ``stop_reason`` defaults to tool_use if any block is a tool call."""
    if stop_reason is None:
        has_tool = any(isinstance(b, ToolUseBlock) for b in content)
        stop_reason = StopReason.TOOL_USE if has_tool else StopReason.END_TURN
    return ModelResponse(
        id=f"msg_{next(_ids):04d}",
        provider=Provider.REPLAY,
        model=model,
        content=list(content),
        stop_reason=stop_reason,
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
        latency_ms=1.0,
    )
