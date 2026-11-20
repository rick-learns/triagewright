from __future__ import annotations

import json
from typing import Any

import pytest

from tests.llm.support import GEMINI, ORDER_TOOL, chat, request
from triagewright.contracts.llm import (
    Message,
    Provider,
    StopReason,
    TextBlock,
    ToolChoice,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)
from triagewright.llm import ModelRequestError, ModelResponseError
from triagewright.llm.chat_wire import (
    from_chat_response,
    to_chat_request,
    tool_use_id,
)

pytestmark = pytest.mark.anyio


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


def test_length_drops_cut_off_tool_calls() -> None:
    req = request(tools=[ORDER_TOOL])
    calls = [
        {"id": "a", "function": {"name": "get_order", "arguments": '{"order_id":"ord_a1b2"}'}},
        {"id": "b", "function": {"name": "get_order", "arguments": '{"order_id":"ord_'}},
        {"id": "c", "function": {"name": "get_order", "arguments": ""}},
        {"id": "d", "function": {"name": "get_order"}},
        {"id": "e", "function": {"name": "list_orders", "arguments": {}}},
        {"id": "f", "function": {"name": "list_orders", "arguments": []}},
        {"id": "g", "function": {"name": "list_orders", "arguments": False}},
        {"id": "h", "function": {"name": "list_orders", "arguments": "[]"}},
    ]
    body = chat({"content": "", "tool_calls": calls}, finish="length")
    resp = from_chat_response(body, request=req, provider=Provider.GEMINI, latency_ms=0)
    assert resp.stop_reason is StopReason.MAX_TOKENS
    assert resp.content == [
        ToolUseBlock(id=tool_use_id(req, 0), name="get_order", input={"order_id": "ord_a1b2"}),
        ToolUseBlock(id=tool_use_id(req, 4), name="list_orders", input={}),
    ]


@pytest.mark.parametrize("native", ["length", "max_tokens", "MAX_TOKENS"])
def test_a_native_token_limit_is_max_tokens_even_when_finish_says_tool_calls(native: str) -> None:
    # OpenRouter, Nemotron: reasoning used up max_tokens mid-arguments, reported as tool_calls.
    req = request(tools=[ORDER_TOOL])
    calls = [{"id": "a", "function": {"name": "get_order", "arguments": '{"order_id": "'}}]
    body = chat({"content": "", "tool_calls": calls}, finish="tool_calls")
    body["choices"][0]["native_finish_reason"] = native
    resp = from_chat_response(body, request=req, provider=Provider.OPENROUTER, latency_ms=0)
    assert resp.stop_reason is StopReason.MAX_TOKENS
    assert resp.content == []


@pytest.mark.parametrize(
    "body",
    [
        chat({"role": "assistant", "content": "x"}, finish="error"),
        chat({"role": "assistant", "content": "x"}, finish=None),
        {"choices": []},
        {"choices": [{"finish_reason": "stop"}]},
        chat({"tool_calls": [{"function": {"name": "f", "arguments": "{not json"}}]}),
        chat({"tool_calls": [{"function": {"name": "f", "arguments": "[1]"}}]}),
        chat({"tool_calls": [{"function": {"name": "f", "arguments": []}}]}),
        chat({"tool_calls": [{"function": {"name": "f", "arguments": False}}]}),
        chat({"tool_calls": [{"function": {"arguments": "{}"}}]}),
    ],
)
def test_malformed_responses_raise(body: dict[str, Any]) -> None:
    with pytest.raises(ModelResponseError):
        from_chat_response(body, request=request(), provider=Provider.GEMINI, latency_ms=0)


def test_usage_follows_anthropic_semantics_and_cost_only_when_reported() -> None:
    usage = {
        "prompt_tokens": 170,
        "completion_tokens": 30,
        "prompt_tokens_details": {"cached_tokens": 100, "cache_write_tokens": 50},
        "cost": 0,
    }
    message = {"role": "assistant", "content": [{"type": "text", "text": "hi"}]}
    resp = from_chat_response(
        chat(message, usage=usage), request=request(), provider=Provider.OPENROUTER, latency_ms=0
    )
    assert resp.usage == Usage(
        input_tokens=20,
        output_tokens=30,
        cache_read_input_tokens=100,
        cache_creation_input_tokens=50,
    )
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


def test_odd_signature_shapes_are_ignored() -> None:
    from triagewright.llm.openai_compat import thought_signatures

    call = {"function": {"name": "f", "arguments": "{}"}, "extra_content": "not-an-object"}
    assert thought_signatures(chat({"tool_calls": [call]}), request()) == {}
    assert thought_signatures({"choices": []}, request()) == {}
