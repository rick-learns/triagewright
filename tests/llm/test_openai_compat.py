from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from tests.llm.support import ORDER_TOOL, chat, gemini, request
from triagewright.contracts.llm import (
    Message,
    ModelResponse,
    StopReason,
    ToolResultBlock,
    ToolUseBlock,
)
from triagewright.llm.openai_compat import to_chat_request

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def test_gemini_resends_thought_signatures_and_falls_back_to_the_sentinel() -> None:
    sent: list[dict[str, Any]] = []
    calls = [
        {
            "id": "function-call-1",
            "type": "function",
            "function": {"name": "get_order", "arguments": '{"order_id":"ord_a1b2"}'},
            "extra_content": {"google": {"thought_signature": "SIG_A"}},
        },
        {
            "id": "function-call-2",
            "type": "function",
            "function": {"name": "get_order", "arguments": '{"order_id":"ord_c3d4"}'},
        },
    ]
    replies = iter(
        [
            chat({"role": "assistant", "content": None, "tool_calls": calls}, "tool_calls"),
            chat({"role": "assistant", "content": "Both shipped."}),
        ]
    )

    def handler(req: httpx.Request) -> httpx.Response:
        sent.append(json.loads(req.content))
        return httpx.Response(200, json=next(replies))

    client = gemini(handler)
    first_req = request(tools=[ORDER_TOOL])
    first = await client.complete(first_req)
    a, b = (blk for blk in first.content if isinstance(blk, ToolUseBlock))
    follow = request(
        tools=[ORDER_TOOL],
        messages=[
            *first_req.messages,
            Message(role="assistant", content=first.content),
            Message(
                role="user",
                content=[
                    ToolResultBlock(tool_use_id=a.id, content="{}"),
                    ToolResultBlock(tool_use_id=b.id, content="{}"),
                ],
            ),
        ],
    )
    assert (await client.complete(follow)).stop_reason is StopReason.END_TURN
    wire_calls = sent[1]["messages"][2]["tool_calls"]
    assert [c["id"] for c in wire_calls] == [a.id, b.id]
    assert wire_calls[0]["extra_content"] == {"google": {"thought_signature": "SIG_A"}}
    assert "extra_content" not in wire_calls[1]
    assert follow.fingerprint() != first_req.fingerprint()  # signatures never enter fingerprints

    # History this client didn't produce (a resumed recording): the sentinel on the first call.
    fresh = gemini(lambda req: httpx.Response(200, json=chat({"content": "ok"})))
    body = fresh._wire_body(follow)
    google = {"google": {"thought_signature": "skip_thought_signature_validator"}}
    assert body["messages"][2]["tool_calls"][0]["extra_content"] == google
    assert "extra_content" not in body["messages"][2]["tool_calls"][1]
    # Other providers never get Gemini's extension.
    assert "extra_content" not in json.dumps(to_chat_request(follow))


def signed_call(args: str, sig: str) -> dict[str, Any]:
    return {
        "id": "x",
        "function": {"name": "get_order", "arguments": args},
        "extra_content": {"google": {"thought_signature": sig}},
    }


async def test_gemini_signatures_follow_call_content_not_just_ids() -> None:
    # Identical requests get identical tool-use ids; each reply keeps its own signature,
    # and a call two replies signed differently falls back to the sentinel.
    replies = iter(
        [
            [signed_call('{"order_id":"ord_a1b2"}', "SIG_A")],
            [signed_call('{"order_id":"ord_c3d4"}', "SIG_B")],
            [signed_call('{"order_id":"ord_a1b2"}', "SIG_C")],
        ]
    )
    client = gemini(
        lambda req: httpx.Response(200, json=chat({"tool_calls": next(replies)}, "tool_calls"))
    )
    req = request(tools=[ORDER_TOOL])
    first, second = await client.complete(req), await client.complete(req)
    assert first.content[0].id == second.content[0].id  # type: ignore[union-attr]

    def sent_signature(resp: ModelResponse) -> object:
        follow = request(messages=[*req.messages, Message(role="assistant", content=resp.content)])
        call = client._wire_body(follow)["messages"][2]["tool_calls"][0]
        return call["extra_content"]["google"]["thought_signature"]

    assert sent_signature(first) == "SIG_A" and sent_signature(second) == "SIG_B"
    await client.complete(req)  # ord_a1b2 again, signed SIG_C
    assert sent_signature(first) == "skip_thought_signature_validator"
    assert sent_signature(second) == "SIG_B"


async def test_gemini_signature_map_is_bounded() -> None:
    n = 0

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal n
        n += 1
        return httpx.Response(200, json=chat({"tool_calls": [signed_call("{}", f"S{n}")]}))

    client = gemini(handler)
    client.max_signatures = 2
    for i in range(3):
        await client.complete(request(system=f"s{i}"))
    assert list(client._signatures.values()) == ["S2", "S3"]
