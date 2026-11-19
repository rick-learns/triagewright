"""Translation between the Messages-shaped contracts and the chat-completions wire format.

``ModelRequest`` to the wire and the wire back to ``ModelResponse``:

- ``system`` becomes a leading system message. In a user turn, each
  ``tool_result`` becomes a ``tool`` message (placed first, because tool
  messages must directly follow the assistant's ``tool_calls``), then the text
  blocks become one user message. An assistant turn becomes one message with
  its text and ``tool_calls`` (arguments as ``canonical_json``).
- Tool-call ids are ``stable_id("toolu", request fingerprint, index)``, not the
  provider's (often random) ids, so identical model output yields identical
  follow-up fingerprints.
- ``finish_reason``: ``stop`` -> end_turn (tool_use if the message has tool
  calls, since some providers report ``stop`` for them), ``tool_calls`` ->
  tool_use, ``length`` -> max_tokens, ``content_filter`` -> refusal. Anything
  else is a :class:`ModelResponseError`. On ``length``, tool calls whose
  arguments were cut off are dropped, so the caller still sees max_tokens.
- Usage follows the contract: ``input_tokens`` excludes cache reads and writes.
  ``provider_cost_usd`` is set only when the provider reports ``usage.cost``.
  Calls don't stream, so ``ttft_ms`` is not measurable and stays ``None``.
- Gemini thought signatures (``tool_calls[].extra_content.google``) are read
  from responses with :func:`thought_signatures` and written back through the
  ``signatures`` lookup of :func:`to_chat_request`.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from triagewright.contracts.common import canonical_json, stable_id
from triagewright.contracts.llm import (
    ContentBlock,
    Message,
    ModelRequest,
    ModelResponse,
    Provider,
    StopReason,
    TextBlock,
    ToolChoice,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)
from triagewright.llm.errors import ModelRequestError, ModelResponseError

_STOP_REASONS = {
    "stop": StopReason.END_TURN,
    "tool_calls": StopReason.TOOL_USE,
    "length": StopReason.MAX_TOKENS,
    "content_filter": StopReason.REFUSAL,
}
SKIP_THOUGHT_SIGNATURE = "skip_thought_signature_validator"

type SignatureLookup = Callable[[ToolUseBlock], str | None]
type SignatureKey = tuple[str, str, str]


def tool_use_id(request: ModelRequest, index: int) -> str:
    return stable_id("toolu", request.fingerprint(), index)


def signature_key(block: ToolUseBlock) -> SignatureKey:
    # Identical requests reuse tool-use ids, so the call's content is part of the key.
    return block.id, block.name, canonical_json(block.input)


def to_chat_request(
    request: ModelRequest, *, signatures: SignatureLookup | None = None
) -> dict[str, Any]:
    """The chat-completions body for ``request`` (``purpose`` is never sent).

    With ``signatures`` (Gemini), each assistant tool call gets its thought
    signature, and the first call of a turn gets the skip sentinel if unknown.
    """
    messages: list[Any] = []
    if request.system:
        messages.append({"role": "system", "content": request.system})
    for message in request.messages:
        messages.extend(_wire_messages(message, signatures))
    body: dict[str, Any] = {
        "model": request.model,
        "messages": messages,
        "max_tokens": request.max_tokens,
    }
    if request.temperature is not None:
        body["temperature"] = request.temperature
    if request.tools:
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in request.tools
        ]
        if request.tool_choice is not None:
            body["tool_choice"] = _wire_tool_choice(request.tool_choice)
    return body


def _wire_messages(message: Message, signatures: SignatureLookup | None) -> list[dict[str, Any]]:
    texts = [b.text for b in message.content if isinstance(b, TextBlock)]
    if message.role == "assistant":
        calls: list[dict[str, Any]] = []
        for b in (b for b in message.content if isinstance(b, ToolUseBlock)):
            call: dict[str, Any] = {
                "id": b.id,
                "type": "function",
                "function": {"name": b.name, "arguments": canonical_json(b.input)},
            }
            if signatures is not None:
                sig = signatures(b) or (None if calls else SKIP_THOUGHT_SIGNATURE)
                if sig:
                    call["extra_content"] = {"google": {"thought_signature": sig}}
            calls.append(call)
        out: dict[str, Any] = {"role": "assistant", "content": "\n\n".join(texts) or None}
        if calls:
            out["tool_calls"] = calls
        return [out]
    wire: list[dict[str, Any]] = [
        {
            "role": "tool",
            "tool_call_id": b.tool_use_id,
            "content": f"[tool error] {b.content}" if b.is_error else b.content,
        }
        for b in message.content
        if isinstance(b, ToolResultBlock)
    ]
    if texts:
        wire.append({"role": "user", "content": "\n\n".join(texts)})
    return wire


def _wire_tool_choice(choice: ToolChoice) -> str | dict[str, Any]:
    if choice.type == "tool":
        if not choice.name:
            raise ModelRequestError("tool_choice 'tool' needs a name")
        return {"type": "function", "function": {"name": choice.name}}
    return {"auto": "auto", "any": "required", "none": "none"}[choice.type]


def from_chat_response(
    body: Any, *, request: ModelRequest, provider: Provider, latency_ms: float
) -> ModelResponse:
    """Translate a chat-completions body. Raises :class:`ModelResponseError` if malformed."""
    try:
        choice = body["choices"][0]
        message = choice["message"]
        finish = choice.get("finish_reason")
        content: list[ContentBlock] = []
        text = _message_text(message.get("content"))
        if text:
            content.append(TextBlock(text=text))
        for i, call in enumerate(message.get("tool_calls") or []):
            fn = call["function"]
            raw = fn.get("arguments")
            if raw is None or raw == "":
                if finish == "length":
                    continue  # cut off before any arguments; the caller sees max_tokens
                raw = "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else raw
            except ValueError:
                if finish == "length":
                    continue  # cut off mid-arguments; the caller sees max_tokens
                raise
            if not isinstance(args, dict):
                if finish == "length":
                    continue  # not a complete object; the caller sees max_tokens
                raise TypeError("tool arguments are not an object")
            content.append(ToolUseBlock(id=tool_use_id(request, i), name=fn["name"], input=args))
        usage, cost = _usage(body.get("usage") or {})
        msg_id = body.get("id")
    except (KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
        raise ModelResponseError(
            f"{provider} {request.model}: malformed response ({type(exc).__name__})"
        ) from exc
    has_tools = any(isinstance(b, ToolUseBlock) for b in content)
    if finish not in _STOP_REASONS:
        raise ModelResponseError(f"{provider} {request.model}: unknown finish_reason {finish!r}")
    stop = _STOP_REASONS[finish]
    if stop is StopReason.END_TURN and has_tools:
        stop = StopReason.TOOL_USE
    return ModelResponse(
        id=msg_id
        if isinstance(msg_id, str) and msg_id
        else stable_id("msg", request.fingerprint()),
        provider=provider,
        model=request.model,
        content=content,
        stop_reason=stop,
        usage=usage,
        latency_ms=latency_ms,
        provider_cost_usd=cost,
    )


def _message_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    # Some providers return a list of content parts.
    return "".join(p["text"] for p in content if p.get("type") == "text")


def thought_signatures(body: Any, request: ModelRequest) -> dict[str, str]:
    """Gemini thought signatures in a chat-completions body, by our tool-use id."""
    found: dict[str, str] = {}
    try:
        calls = body["choices"][0]["message"].get("tool_calls") or []
        for i, call in enumerate(calls):
            sig = ((call.get("extra_content") or {}).get("google") or {}).get("thought_signature")
            if isinstance(sig, str) and sig:
                found[tool_use_id(request, i)] = sig
    except (KeyError, IndexError, TypeError, AttributeError):
        return {}
    return found


def _usage(raw: Any) -> tuple[Usage, float | None]:
    prompt = int(raw.get("prompt_tokens") or 0)
    details = raw.get("prompt_tokens_details") or {}
    cached = int(details.get("cached_tokens") or 0)
    written = int(details.get("cache_write_tokens") or 0)
    usage = Usage(
        input_tokens=max(prompt - cached - written, 0),
        output_tokens=int(raw.get("completion_tokens") or 0),
        cache_read_input_tokens=cached,
        cache_creation_input_tokens=written,
    )
    cost = raw.get("cost")
    is_number = isinstance(cost, int | float) and not isinstance(cost, bool)
    return usage, float(cost) if is_number else None
