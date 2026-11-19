from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

import httpx
import pytest

from tests.llm.support import GEMINI, KEY, Sleeps, chat, gemini, request
from triagewright.contracts.llm import Provider, StopReason, TextBlock
from triagewright.llm import (
    GeminiClient,
    ModelAuthError,
    ModelConfigError,
    ModelRequestError,
    ModelResponseError,
    ModelUnavailableError,
    OpenRouterClient,
)
from triagewright.llm.chat_transport import retry_after_seconds
from triagewright.llm.chat_wire import to_chat_request
from triagewright.llm.ratelimit import RateLimiter
from triagewright.pacing import CallBudget

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


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
        err = {"message": "q", "status": "RESOURCE_EXHAUSTED"}
        return httpx.Response(429, headers={"Retry-After": "3"}, json={"error": err})

    with pytest.raises(ModelUnavailableError, match="HTTP 429 RESOURCE_EXHAUSTED") as info:
        await gemini(handler, sleeps).complete(request())
    assert calls == 3 and sleeps.calls == [3.0, 3.0]
    assert info.value.retryable and info.value.status == 429 and info.value.retry_after_s == 3.0


OK = {"role": "assistant", "content": "ok"}


def gemini_quota_429(delay: str) -> httpx.Response:
    # The shape Gemini's OpenAI-compatible endpoint returns: a list, details in the body.
    details = [
        {
            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
            "violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}],
        },
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay},
    ]
    err = {"code": 429, "status": "RESOURCE_EXHAUSTED", "details": details}
    return httpx.Response(429, json=[{"error": err}])


async def test_gemini_retry_info_in_the_body_sets_the_delay() -> None:
    sleeps = Sleeps()
    responses = iter([gemini_quota_429("13s"), httpx.Response(200, json=chat(OK))])
    response = await gemini(lambda req: next(responses), sleeps).complete(request())
    assert response.stop_reason and sleeps.calls == [13.0]
    with pytest.raises(ModelUnavailableError) as info:
        await gemini(lambda req: gemini_quota_429("45.5s"), Sleeps()).complete(request())
    assert info.value.retry_after_s == 45.5  # beyond the 30s cap: fail fast
    sleeps = Sleeps()
    odd = iter([gemini_quota_429("soon"), httpx.Response(200, json=chat(OK))])
    await gemini(lambda req: next(odd), sleeps).complete(request())
    assert sleeps.calls == [0.5]  # unparsable: the computed backoff


async def test_a_limiter_paces_every_attempt() -> None:
    clock = [0.0]

    class AdvancingSleeps(Sleeps):
        async def __call__(self, seconds: float) -> None:
            await super().__call__(seconds)
            clock[0] += seconds

    sleeps = AdvancingSleeps()
    limiter = RateLimiter(1, window_s=60, clock=lambda: clock[0], sleep=sleeps)
    responses = iter([httpx.Response(503), httpx.Response(200, json=chat(OK))])
    http = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: next(responses)))
    client = GeminiClient(
        model=GEMINI,
        api_key=KEY,
        base_url="https://gemini.test/v1beta/openai/",
        http_client=http,
        sleep=sleeps,
        limiter=limiter,
    )
    response = await client.complete(request())
    assert sleeps.calls == [0.5, 59.5]  # the backoff, then the retry waits for a slot
    assert response.stop_reason is StopReason.END_TURN


async def test_a_retry_that_cannot_fit_the_deadline_fails_at_once() -> None:
    async def complete(timeout_s: float) -> list[float]:
        sleeps = Sleeps()
        responses = iter([gemini_quota_429("13s"), httpx.Response(200, json=chat(OK))])
        client = gemini(lambda req: next(responses), sleeps)
        async with asyncio.timeout(timeout_s) as deadline:
            with CallBudget(deadline).installed():
                await client.complete(request())
        return sleeps.calls

    with pytest.raises(ModelUnavailableError, match="HTTP 429"):
        await complete(10)  # 13 s + another attempt don't fit in 10 s
    assert await complete(20) == [13.0]


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
        # The provider's free text can echo the request; only its status code is kept.
        err = {"message": f"bad key {KEY} for 'Where is ord_a1b2?'", "status": "INVALID_ARGUMENT"}
        return httpx.Response(status, json=[{"error": err}])

    with pytest.raises(error) as info:
        await gemini(handler, sleeps).complete(request())
    assert calls == 1 and sleeps.calls == []
    assert str(info.value) == f"gemini {GEMINI}: HTTP {status} INVALID_ARGUMENT"


async def test_openrouter_choice_that_finished_with_error_is_retried() -> None:
    bodies = iter([chat({"content": ""}, finish="error"), chat({"content": "ok"})])
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
    assert resp.content == [TextBlock(text="ok")]
    assert sleeps.calls == [0.5]


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
    assert retry_after_seconds("Thu Nov 19 12:00:07 2026", now=now) == 7.0
    assert retry_after_seconds("soon") is None
    assert retry_after_seconds(None) is None
