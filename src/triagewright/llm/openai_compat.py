"""One OpenAI-compatible chat-completions layer behind the Gemini and OpenRouter clients.

Translation (``ModelRequest`` to the wire and the wire back to ``ModelResponse``):

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
  else is a :class:`ModelResponseError`.
- Usage follows the contract: ``input_tokens`` excludes cached prompt tokens.
  ``provider_cost_usd`` is set only when the provider reports ``usage.cost``.
  Calls don't stream, so ``ttft_ms`` is not measurable and stays ``None``.

Retry policy (the only retries; the port forbids silent ones beyond it): 408,
429, 500, 502, 503, 504, timeouts, and connection errors are retried up to
``max_retries`` times with exponential backoff (``backoff_s`` doubling, capped
at ``max_delay_s``). A ``Retry-After`` header replaces the computed delay; one
longer than ``max_delay_s`` fails at once rather than stall the turn. After the
budget the client raises :class:`ModelUnavailableError`. 401/403 raise
:class:`ModelAuthError` and other 4xx :class:`ModelRequestError`, without
retrying. ``latency_ms`` is the wall time of the whole call, retries included.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, ClassVar

import httpx

from triagewright.config import is_free_model
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
from triagewright.llm.errors import (
    ModelAuthError,
    ModelConfigError,
    ModelError,
    ModelRequestError,
    ModelResponseError,
    ModelUnavailableError,
)

log = logging.getLogger(__name__)

RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})
_STOP_REASONS = {
    "stop": StopReason.END_TURN,
    "tool_calls": StopReason.TOOL_USE,
    "function_call": StopReason.TOOL_USE,
    "length": StopReason.MAX_TOKENS,
    "content_filter": StopReason.REFUSAL,
}
_ERROR_TEXT_LIMIT = 200

type Sleep = Callable[[float], Awaitable[None]]


def tool_use_id(request: ModelRequest, index: int) -> str:
    return stable_id("toolu", request.fingerprint(), index)


def to_chat_request(request: ModelRequest) -> dict[str, Any]:
    """The chat-completions body for ``request`` (``purpose`` is never sent)."""
    messages: list[Any] = []
    if request.system:
        messages.append({"role": "system", "content": request.system})
    for message in request.messages:
        messages.extend(_wire_messages(message))
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


def _wire_messages(message: Message) -> list[dict[str, Any]]:
    texts = [b.text for b in message.content if isinstance(b, TextBlock)]
    if message.role == "assistant":
        calls = [
            {
                "id": b.id,
                "type": "function",
                "function": {"name": b.name, "arguments": canonical_json(b.input)},
            }
            for b in message.content
            if isinstance(b, ToolUseBlock)
        ]
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
            raw = fn.get("arguments") or "{}"
            args = json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(args, dict):
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


def _usage(raw: Any) -> tuple[Usage, float | None]:
    prompt = int(raw.get("prompt_tokens") or 0)
    details = raw.get("prompt_tokens_details") or {}
    cached = int(details.get("cached_tokens") or 0)
    usage = Usage(
        input_tokens=max(prompt - cached, 0),
        output_tokens=int(raw.get("completion_tokens") or 0),
        cache_read_input_tokens=cached,
    )
    cost = raw.get("cost")
    is_number = isinstance(cost, int | float) and not isinstance(cost, bool)
    return usage, float(cost) if is_number else None


def retry_after_seconds(value: str | None, *, now: datetime | None = None) -> float | None:
    """Parse ``Retry-After`` (seconds or an HTTP date). ``None`` if absent or unparsable."""
    if not value:
        return None
    try:
        return max(float(value), 0.0)
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max((when - (now or datetime.now(UTC))).total_seconds(), 0.0)


class OpenAICompatClient:
    """A ``ports.ModelClient`` for one free model on an OpenAI-compatible endpoint."""

    provider: ClassVar[Provider]

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        http_client: httpx.AsyncClient,
        max_retries: int = 2,
        backoff_s: float = 0.5,
        max_delay_s: float = 10.0,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        if not is_free_model(self.provider, model):
            raise ModelConfigError(f"{model!r} is not a free {self.provider} model")
        if not api_key:
            raise ModelConfigError(f"{self.provider} needs an API key")
        self.model = model
        self._api_key = api_key
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._http = http_client
        self._max_retries = max_retries
        self._backoff_s = backoff_s
        self._max_delay_s = max_delay_s
        self._sleep = sleep

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r})"

    async def complete(self, request: ModelRequest) -> ModelResponse:
        if request.model != self.model:
            raise ModelConfigError(f"client for {self.model!r} got a request for {request.model!r}")
        body = to_chat_request(request)
        started = time.perf_counter()
        attempt = 0
        while True:
            try:
                data = await self._post(body)
            except ModelUnavailableError as exc:
                delay = self._delay(attempt, exc.retry_after_s)
                if attempt >= self._max_retries or delay is None:
                    raise
                attempt += 1
                log.warning(
                    "%s %s unavailable (%s); retry %d/%d in %.2fs",
                    self.provider,
                    self.model,
                    exc.status or "network",
                    attempt,
                    self._max_retries,
                    delay,
                )
                await self._sleep(delay)
                continue
            latency_ms = (time.perf_counter() - started) * 1000
            return from_chat_response(
                data, request=request, provider=self.provider, latency_ms=latency_ms
            )

    def _delay(self, attempt: int, retry_after_s: float | None) -> float | None:
        delay = (
            retry_after_s
            if retry_after_s is not None
            else min(self._backoff_s * 2**attempt, self._max_delay_s)
        )
        return delay if delay <= self._max_delay_s else None

    async def _post(self, body: dict[str, Any]) -> Any:
        where = f"{self.provider} {self.model}"
        try:
            resp = await self._http.post(
                self._url, json=body, headers={"Authorization": f"Bearer {self._api_key}"}
            )
        except httpx.TimeoutException as exc:
            raise ModelUnavailableError(f"{where}: timed out ({type(exc).__name__})") from exc
        except httpx.TransportError as exc:
            raise ModelUnavailableError(
                f"{where}: connection failed ({type(exc).__name__})"
            ) from exc
        data = _json_or_none(resp)
        status = resp.status_code
        # OpenRouter can report an upstream failure as a 200 with an ``error`` body.
        if status == 200 and isinstance(data, dict) and "error" in data and not data.get("choices"):
            code = data["error"].get("code") if isinstance(data["error"], dict) else None
            status = code if isinstance(code, int) and code >= 400 else 502
        if status < 400:
            if data is None:
                raise ModelResponseError(f"{where}: response is not JSON")
            return data
        raise _status_error(where, status, data, resp.headers.get("retry-after"))


def _json_or_none(resp: httpx.Response) -> Any:
    try:
        return resp.json()
    except ValueError:
        return None


def _status_error(where: str, status: int, data: Any, retry_after: str | None) -> ModelError:
    if isinstance(data, list) and data:  # Gemini sometimes wraps the error body in a list
        data = data[0]
    err = data.get("error") if isinstance(data, dict) else None
    detail = ""
    if isinstance(err, dict) and isinstance(err.get("message"), str):
        detail = ": " + err["message"][:_ERROR_TEXT_LIMIT]
    message = f"{where}: HTTP {status}{detail}"
    if status in RETRYABLE_STATUS:
        return ModelUnavailableError(
            message, status=status, retry_after_s=retry_after_seconds(retry_after)
        )
    if status in (401, 403):
        return ModelAuthError(message, status=status)
    if 400 <= status < 500:
        return ModelRequestError(message, status=status)
    return ModelResponseError(message, status=status)


class GeminiClient(OpenAICompatClient):
    """Gemini API free tier through its OpenAI-compatible endpoint."""

    provider = Provider.GEMINI


class OpenRouterClient(OpenAICompatClient):
    """OpenRouter, ``:free`` models only."""

    provider = Provider.OPENROUTER
