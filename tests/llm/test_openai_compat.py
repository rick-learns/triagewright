from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from triagewright.contracts.llm import (
    Message,
    ModelRequest,
    Provider,
    StopReason,
    TextBlock,
    ToolChoice,
    ToolDefinition,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)
from triagewright.llm import (
    GeminiClient,
    ModelAuthError,
    ModelConfigError,
    ModelRequestError,
    ModelResponseError,
    ModelUnavailableError,
    OpenRouterClient,
)
from triagewright.llm.openai_compat import (
    from_chat_response,
    retry_after_seconds,
    to_chat_request,
    tool_use_id,
)

pytestmark = pytest.mark.anyio

GEMINI = "gemini-3.5-flash-lite"
KEY = "test-key-not-real"
ORDER_TOOL = ToolDefinition(
    name="get_order",
    description="Look up an order.",
    input_schema={"type": "object", "properties": {"order_id": {"type": "string"}}},
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


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


def test_multi_tool_turn_translates_to_wire() -> None:
    a, b = tool_use_id(request(), 0), tool_use_id(request(), 1)
    req = request(
        messages=[
            Message(role="user", content=[TextBlock(text="Refund ord_a1b2 and ord_c3d4")]),
            Message(
                role="assistant",
                content=[
                    TextBlock(text="Checking both."),
                    ToolUseBlock(id=a, name="get_order", input={"order_id": "ord_a1b2"}),
                    ToolUseBlock(id=b, name="get_order", input={"order_id": "ord_c3d4"}),
                ],
            ),
            Message(
                role="user",
                content=[
                    TextBlock(text="Thanks."),
                    ToolResultBlock(tool_use_id=a, content='{"ok":true}'),
                    ToolResultBlock(tool_use_id=b, content='{"isError":true}', is_error=True),
                ],
            ),
        ],
        tools=[ORDER_TOOL],
        tool_choice=ToolChoice(type="any"),
        max_tokens=256,
        temperature=0.2,
    )
    body = to_chat_request(req)
    assert body == {
        "model": GEMINI,
        "max_tokens": 256,
        "temperature": 0.2,
        "tool_choice": "required",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_order",
                    "description": "Look up an order.",
                    "parameters": ORDER_TOOL.input_schema,
                },
            }
        ],
        "messages": [
            {"role": "system", "content": "You are support."},
            {"role": "user", "content": "Refund ord_a1b2 and ord_c3d4"},
            {
                "role": "assistant",
                "content": "Checking both.",
                "tool_calls": [
                    {
                        "id": a,
                        "type": "function",
                        "function": {"name": "get_order", "arguments": '{"order_id":"ord_a1b2"}'},
                    },
                    {
                        "id": b,
                        "type": "function",
                        "function": {"name": "get_order", "arguments": '{"order_id":"ord_c3d4"}'},
                    },
                ],
            },
            # Tool results come straight after the tool calls, then the user's text.
            {"role": "tool", "tool_call_id": a, "content": '{"ok":true}'},
            {"role": "tool", "tool_call_id": b, "content": '[tool error] {"isError":true}'},
            {"role": "user", "content": "Thanks."},
        ],
    }
    assert "purpose" not in json.dumps(body)


@pytest.mark.parametrize(
    ("choice", "wire"),
    [
        (ToolChoice(type="auto"), "auto"),
        (ToolChoice(type="none"), "none"),
        (ToolChoice(type="any"), "required"),
        (
            ToolChoice(type="tool", name="finalize"),
            {"type": "function", "function": {"name": "finalize"}},
        ),
    ],
)
def test_tool_choice_mapping(choice: ToolChoice, wire: object) -> None:
    assert to_chat_request(request(tools=[ORDER_TOOL], tool_choice=choice))["tool_choice"] == wire


def test_optional_fields_are_omitted() -> None:
    body = to_chat_request(
        request(system="", temperature=None, tool_choice=ToolChoice(type="auto"))
    )
    assert set(body) == {"model", "messages", "max_tokens"}
    assert body["messages"] == [{"role": "user", "content": "Where is ord_a1b2?"}]
    tool_only = Message(role="assistant", content=[ToolUseBlock(id="t", name="x", input={})])
    assert to_chat_request(request(messages=[*request().messages, tool_only]))["messages"][-1] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "t", "type": "function", "function": {"name": "x", "arguments": "{}"}}
        ],
    }
    with pytest.raises(ModelRequestError, match="needs a name"):
        to_chat_request(request(tools=[ORDER_TOOL], tool_choice=ToolChoice(type="tool")))


def test_multi_tool_response_round_trips_into_the_next_request() -> None:
    req = request(tools=[ORDER_TOOL])
    body = chat(
        {
            "role": "assistant",
            "content": "Let me look.",
            "tool_calls": [
                {
                    "id": "provider-random-1",
                    "type": "function",
                    "function": {"name": "get_order", "arguments": '{"order_id": "ord_a1b2"}'},
                },
                {
                    "id": "",
                    "type": "function",
                    "function": {"name": "list_orders", "arguments": ""},
                },
            ],
        },
        finish="tool_calls",
    )
    resp = from_chat_response(body, request=req, provider=Provider.GEMINI, latency_ms=12.5)
    assert resp.stop_reason is StopReason.TOOL_USE
    assert resp.content == [
        TextBlock(text="Let me look."),
        ToolUseBlock(id=tool_use_id(req, 0), name="get_order", input={"order_id": "ord_a1b2"}),
        ToolUseBlock(id=tool_use_id(req, 1), name="list_orders", input={}),
    ]
    assert resp.id == "chatcmpl-1" and resp.provider is Provider.GEMINI and resp.model == GEMINI
    assert resp.latency_ms == 12.5 and resp.ttft_ms is None and resp.provider_cost_usd is None
    # Ids are deterministic: the same response to the same request maps the same way.
    again = from_chat_response(body, request=req, provider=Provider.GEMINI, latency_ms=1)
    assert [b.id for b in again.content[1:]] == [b.id for b in resp.content[1:]]  # type: ignore[union-attr]
    # Feeding the assistant turn back reproduces the provider's tool calls under our ids.
    follow = request(messages=[*req.messages, Message(role="assistant", content=resp.content)])
    calls = to_chat_request(follow)["messages"][-1]["tool_calls"]
    assert [c["id"] for c in calls] == [tool_use_id(req, 0), tool_use_id(req, 1)]


@pytest.mark.parametrize(
    ("finish", "tool_calls", "stop"),
    [
        ("stop", False, StopReason.END_TURN),
        ("stop", True, StopReason.TOOL_USE),  # some providers say "stop" for tool calls
        ("tool_calls", True, StopReason.TOOL_USE),
        ("function_call", True, StopReason.TOOL_USE),
        ("length", False, StopReason.MAX_TOKENS),
        ("length", True, StopReason.MAX_TOKENS),
        ("content_filter", False, StopReason.REFUSAL),
    ],
)
def test_stop_reason_mapping(finish: str, tool_calls: bool, stop: StopReason) -> None:
    message: dict[str, Any] = {"role": "assistant", "content": "x"}
    if tool_calls:
        message["tool_calls"] = [{"id": "c", "function": {"name": "f", "arguments": "{}"}}]
    resp = from_chat_response(
        chat(message, finish), request=request(), provider=Provider.GEMINI, latency_ms=0
    )
    assert resp.stop_reason is stop


@pytest.mark.parametrize(
    "body",
    [
        chat({"role": "assistant", "content": "x"}, finish="error"),
        chat({"role": "assistant", "content": "x"}, finish=None),
        {"choices": []},
        {"choices": [{"finish_reason": "stop"}]},
        chat({"tool_calls": [{"function": {"name": "f", "arguments": "{not json"}}]}),
        chat({"tool_calls": [{"function": {"name": "f", "arguments": "[1]"}}]}),
        chat({"tool_calls": [{"function": {"arguments": "{}"}}]}),
    ],
)
def test_malformed_responses_raise(body: dict[str, Any]) -> None:
    with pytest.raises(ModelResponseError):
        from_chat_response(body, request=request(), provider=Provider.GEMINI, latency_ms=0)


def test_usage_follows_anthropic_semantics_and_cost_only_when_reported() -> None:
    usage = {
        "prompt_tokens": 120,
        "completion_tokens": 30,
        "prompt_tokens_details": {"cached_tokens": 100},
        "cost": 0,
    }
    message = {"role": "assistant", "content": [{"type": "text", "text": "hi"}]}
    resp = from_chat_response(
        chat(message, usage=usage), request=request(), provider=Provider.OPENROUTER, latency_ms=0
    )
    assert resp.usage == Usage(input_tokens=20, output_tokens=30, cache_read_input_tokens=100)
    assert resp.provider_cost_usd == 0.0
    assert resp.content == [TextBlock(text="hi")]
    bare = from_chat_response(
        {"choices": [{"message": {"content": None}, "finish_reason": "stop"}], "usage": None},
        request=request(),
        provider=Provider.GEMINI,
        latency_ms=0,
    )
    assert bare.usage == Usage() and bare.provider_cost_usd is None and bare.content == []
    assert bare.id.startswith("msg_")  # no provider id: a stable one
    for cost in ("0.1", True, None):
        odd = from_chat_response(
            chat(message, usage={"cost": cost}),
            request=request(),
            provider=Provider.OPENROUTER,
            latency_ms=0,
        )
        assert odd.provider_cost_usd is None


async def test_client_posts_with_bearer_key_and_measures_latency() -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json=chat({"role": "assistant", "content": "On its way."}))

    resp = await gemini(handler).complete(request())
    assert resp.content == [TextBlock(text="On its way.")]
    assert resp.stop_reason is StopReason.END_TURN and resp.latency_ms >= 0
    (sent,) = seen
    assert str(sent.url) == "https://gemini.test/v1beta/openai/chat/completions"
    assert sent.headers["authorization"] == f"Bearer {KEY}"
    assert json.loads(sent.content) == to_chat_request(request())


async def test_retryable_statuses_back_off_then_succeed() -> None:
    statuses = iter([429, 503])
    sleeps = Sleeps()

    def handler(req: httpx.Request) -> httpx.Response:
        status = next(statuses, 200)
        if status == 200:
            return httpx.Response(200, json=chat({"content": "ok"}))
        return httpx.Response(status, json={"error": {"message": "high demand", "code": status}})

    resp = await gemini(handler, sleeps).complete(request())
    assert resp.content == [TextBlock(text="ok")]
    assert sleeps.calls == [0.5, 1.0]


async def test_retry_budget_is_bounded_and_honors_retry_after() -> None:
    sleeps = Sleeps()
    calls = 0

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(429, headers={"Retry-After": "3"}, json={"error": {"message": "q"}})

    with pytest.raises(ModelUnavailableError, match="HTTP 429: q") as info:
        await gemini(handler, sleeps).complete(request())
    assert calls == 3 and sleeps.calls == [3.0, 3.0]
    assert info.value.retryable and info.value.status == 429 and info.value.retry_after_s == 3.0


async def test_retry_after_beyond_the_cap_fails_fast() -> None:
    sleeps = Sleeps()

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(503, headers={"Retry-After": "60"})

    with pytest.raises(ModelUnavailableError, match="HTTP 503"):
        await gemini(handler, sleeps).complete(request())
    assert sleeps.calls == []


@pytest.mark.parametrize("exc", [httpx.ReadTimeout("slow"), httpx.ConnectError("refused")])
async def test_network_errors_are_retryable(exc: httpx.TransportError) -> None:
    sleeps = Sleeps()

    def handler(req: httpx.Request) -> httpx.Response:
        raise exc

    with pytest.raises(ModelUnavailableError, match=r"timed out|connection failed"):
        await gemini(handler, sleeps).complete(request())
    assert sleeps.calls == [0.5, 1.0]


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (401, ModelAuthError),
        (403, ModelAuthError),
        (400, ModelRequestError),
        (404, ModelRequestError),
        (422, ModelRequestError),
        (501, ModelResponseError),
    ],
)
async def test_non_retryable_statuses_raise_at_once(status: int, error: type[Exception]) -> None:
    sleeps = Sleeps()
    calls = 0

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status, json=[{"error": {"message": "nope " * 100}}])

    with pytest.raises(error) as info:
        await gemini(handler, sleeps).complete(request())
    assert calls == 1 and sleeps.calls == []
    assert f"HTTP {status}: nope" in str(info.value)
    assert len(str(info.value)) < 260  # provider text is truncated
    assert KEY not in str(info.value) and "ord_a1b2" not in str(info.value)


async def test_openrouter_error_in_a_200_body_is_classified() -> None:
    bodies = iter(
        [
            {"error": {"code": 429, "message": "upstream rate limited"}},
            {"error": "bad gateway"},
            chat({"content": "ok"}),
        ]
    )
    sleeps = Sleeps()
    http = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json=next(bodies)))
    )
    client = OpenRouterClient(
        model="google/gemma-4-31b-it:free",
        api_key=KEY,
        base_url="https://openrouter.test/api/v1",
        http_client=http,
        sleep=sleeps,
    )
    resp = await client.complete(request(model="google/gemma-4-31b-it:free"))
    assert resp.provider is Provider.OPENROUTER and resp.content == [TextBlock(text="ok")]
    assert sleeps.calls == [0.5, 1.0]


async def test_non_json_success_is_a_response_error() -> None:
    client = gemini(lambda req: httpx.Response(200, text="<html>"))
    with pytest.raises(ModelResponseError, match="not JSON"):
        await client.complete(request())


@pytest.mark.parametrize(
    ("cls", "model"),
    [
        (GeminiClient, "gemini-3.5-pro"),
        (GeminiClient, "gemini-flash-latest"),
        (GeminiClient, "google/gemma-4-31b-it:free"),
        (OpenRouterClient, "openai/gpt-5"),
        (OpenRouterClient, "gemini-3.5-flash-lite"),
    ],
)
def test_paid_models_are_refused_at_construction(
    cls: type[GeminiClient | OpenRouterClient], model: str
) -> None:
    with pytest.raises(ModelConfigError, match="not a free"):
        cls(model=model, api_key=KEY, base_url="https://x.test", http_client=httpx.AsyncClient())


async def test_client_refuses_other_models_and_missing_keys() -> None:
    client = gemini(lambda req: httpx.Response(200, json=chat({"content": "x"})))
    with pytest.raises(ModelConfigError, match="got a request for"):
        await client.complete(request(model="gemini-3.8-flash"))
    with pytest.raises(ModelConfigError, match="API key"):
        GeminiClient(model=GEMINI, api_key="", base_url="x", http_client=httpx.AsyncClient())
    assert KEY not in repr(client)


def test_retry_after_parsing() -> None:
    now = datetime(2026, 11, 19, 12, 0, 0, tzinfo=UTC)
    assert retry_after_seconds("2.5") == 2.5
    assert retry_after_seconds("-1") == 0.0
    assert retry_after_seconds("Thu, 19 Nov 2026 12:00:07 GMT", now=now) == 7.0
    assert retry_after_seconds("Thu, 19 Nov 2026 11:00:00 GMT", now=now) == 0.0
    assert retry_after_seconds("soon") is None
    assert retry_after_seconds(None) is None
