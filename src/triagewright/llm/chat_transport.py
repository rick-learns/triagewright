"""HTTP transport, retries, and error mapping for an OpenAI-compatible chat-completions endpoint.

Retry policy (the only retries; the port forbids silent ones beyond it): 408,
429, 500, 502, 503, 504, timeouts, and connection errors are retried up to
``max_retries`` times with exponential backoff (``backoff_s`` doubling, capped
at ``max_delay_s``). A ``Retry-After`` header replaces the computed delay; one
longer than ``max_delay_s`` fails at once rather than stall the turn. After the
budget the client raises :class:`ModelUnavailableError`. 401/403 raise
:class:`ModelAuthError` and other 4xx :class:`ModelRequestError`, without
retrying. Error messages carry only the provider, model, HTTP status, and the
provider's status code (e.g. ``RESOURCE_EXHAUSTED``), never the provider's
free-text message, which can echo the request. ``latency_ms`` is the wall time
of the whole call, retries included.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, ClassVar

import httpx

from triagewright.config import is_free_model
from triagewright.contracts.llm import ModelRequest, ModelResponse, Provider
from triagewright.llm.chat_wire import from_chat_response, to_chat_request
from triagewright.llm.errors import (
    ModelAuthError,
    ModelConfigError,
    ModelError,
    ModelRequestError,
    ModelResponseError,
    ModelUnavailableError,
)
from triagewright.llm.ratelimit import RateLimiter
from triagewright.pacing import CALL_BUDGET

# The name it had before the split, so log configuration and filters still match.
log = logging.getLogger("triagewright.llm.openai_compat")

RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})
_ERROR_CODE = re.compile(r"[A-Z][A-Z0-9_]{0,63}")

type Sleep = Callable[[float], Awaitable[None]]


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
        max_delay_s: float = 30.0,
        sleep: Sleep = asyncio.sleep,
        limiter: RateLimiter | None = None,
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
        self._limiter = limiter

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r})"

    async def complete(self, request: ModelRequest) -> ModelResponse:
        if request.model != self.model:
            raise ModelConfigError(f"client for {self.model!r} got a request for {request.model!r}")
        body = self._wire_body(request)
        started = time.perf_counter()
        paced_s = 0.0
        attempt = 0
        while True:
            if self._limiter is not None:
                waited = await self._limiter.acquire()
                paced_s += waited
                if waited > 0:
                    log.info("%s %s paced %.2fs", self.provider, self.model, waited)
            try:
                data = await self._post(body)
            except ModelUnavailableError as exc:
                delay = self._delay(attempt, exc.retry_after_s)
                if attempt >= self._max_retries or delay is None or not _fits(delay):
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
            latency_ms = max(time.perf_counter() - started - paced_s, 0.0) * 1000
            response = from_chat_response(
                data, request=request, provider=self.provider, latency_ms=latency_ms
            )
            self._remember(data, request, response)
            return response

    def _wire_body(self, request: ModelRequest) -> dict[str, Any]:
        return to_chat_request(request)

    def _remember(self, data: Any, request: ModelRequest, response: ModelResponse) -> None:
        """Hook for provider state carried between calls (Gemini thought signatures)."""

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


# The least time worth leaving a retry after its backoff.
MIN_ATTEMPT_S = 2.0


def _fits(delay: float) -> bool:
    """Whether a retry after ``delay`` still leaves an attempt's time before the deadline."""
    budget = CALL_BUDGET.get()
    deadline = budget.deadline() if budget is not None else None
    if deadline is None:
        return True
    return asyncio.get_running_loop().time() + delay + MIN_ATTEMPT_S <= deadline


_DURATION = re.compile(r"(\d+(?:\.\d+)?)s")


def _retry_info_seconds(err: Any) -> float | None:
    """Gemini's ``google.rpc.RetryInfo`` delay ("13s") from an error body, if any.

    Its 429s carry the delay in the body, not in a ``Retry-After`` header.
    """
    details = err.get("details") if isinstance(err, dict) else None
    for detail in details if isinstance(details, list) else []:
        if isinstance(detail, dict) and str(detail.get("@type", "")).endswith("rpc.RetryInfo"):
            match = _DURATION.fullmatch(str(detail.get("retryDelay", "")))
            if match:
                return float(match.group(1))
    return None


def _json_or_none(resp: httpx.Response) -> Any:
    try:
        return resp.json()
    except ValueError:
        return None


def _status_error(where: str, status: int, data: Any, retry_after: str | None) -> ModelError:
    if isinstance(data, list) and data:  # Gemini sometimes wraps the error body in a list
        data = data[0]
    err = data.get("error") if isinstance(data, dict) else None
    code = err.get("status") if isinstance(err, dict) else None
    detail = f" {code}" if isinstance(code, str) and _ERROR_CODE.fullmatch(code) else ""
    message = f"{where}: HTTP {status}{detail}"
    if status in RETRYABLE_STATUS:
        wait = retry_after_seconds(retry_after)
        if wait is None:
            wait = _retry_info_seconds(err)
        return ModelUnavailableError(message, status=status, retry_after_s=wait)
    if status in (401, 403):
        return ModelAuthError(message, status=status)
    if 400 <= status < 500:
        return ModelRequestError(message, status=status)
    return ModelResponseError(message, status=status)
