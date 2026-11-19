"""Shared request, response, and client builders for the chat-completions tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx

from triagewright.contracts.llm import Message, ModelRequest, TextBlock, ToolDefinition
from triagewright.llm import GeminiClient

GEMINI = "gemini-3.5-flash-lite"
KEY = "test-key-not-real"
ORDER_TOOL = ToolDefinition(
    name="get_order",
    description="Look up an order.",
    input_schema={"type": "object", "properties": {"order_id": {"type": "string"}}},
)


def request(**overrides: Any) -> ModelRequest:
    fields: dict[str, Any] = {
        "model": GEMINI,
        "system": "You are support.",
        "messages": [Message(role="user", content=[TextBlock(text="Where is ord_a1b2?")])],
        "purpose": "coordinator",
    }
    return ModelRequest(**(fields | overrides))


def chat(
    message: dict[str, Any], finish: str | None = "stop", usage: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 2},
    }


class Sleeps:
    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def gemini(
    handler: Callable[[httpx.Request], httpx.Response], sleep: Sleeps | None = None
) -> GeminiClient:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return GeminiClient(
        model=GEMINI,
        api_key=KEY,
        base_url="https://gemini.test/v1beta/openai/",
        http_client=http,
        sleep=sleep or Sleeps(),
    )
