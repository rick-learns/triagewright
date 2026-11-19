from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from triagewright.config import DEFAULT_MODELS, Settings
from triagewright.contracts.llm import Provider
from triagewright.contracts.ports import ModelClient
from triagewright.llm import (
    GeminiClient,
    ModelConfigError,
    OpenRouterClient,
    RecordingClient,
    ReplayClient,
    build_model_clients,
)
from triagewright.llm.ratelimit import RateLimits

pytestmark = pytest.mark.anyio

ENV = ("TW_LLM_PROVIDER", "TW_RECORD", "GEMINI_API_KEY", "OPENROUTER_API_KEY", "TW_MODEL_JUDGE")


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ENV:
        monkeypatch.delenv(var, raising=False)


def settings(**values: object) -> Settings:
    return Settings(_env_file=None, **values)  # type: ignore[arg-type]


async def test_replay_is_the_default_and_needs_a_case() -> None:
    clients = build_model_clients(settings(), case_id="happy-001")
    assert set(clients.clients) == set(DEFAULT_MODELS[Provider.REPLAY])
    assert all(isinstance(c, ReplayClient) for c in clients.clients.values())
    assert all(isinstance(c, ModelClient) for c in clients.clients.values())
    assert clients.models == DEFAULT_MODELS[Provider.GEMINI]
    await clients.aclose()
    with pytest.raises(ModelConfigError, match="needs a case id"):
        build_model_clients(settings())
    with pytest.raises(ModelConfigError, match="needs a live provider"):
        build_model_clients(settings(record=True), case_id="happy-001")


async def test_live_providers_use_their_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    monkeypatch.setenv("TW_MODEL_JUDGE", "gemma-4-27b-it")
    clients = build_model_clients(settings(llm_provider=Provider.GEMINI))
    judge = clients.for_role("judge")
    assert isinstance(judge, GeminiClient) and judge.model == "gemma-4-27b-it"
    assert clients.for_role("triage").model == "gemini-3.5-flash-lite"  # type: ignore[attr-defined]
    await clients.aclose()

    monkeypatch.delenv("TW_MODEL_JUDGE")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    http = httpx.AsyncClient()
    clients = build_model_clients(settings(llm_provider=Provider.OPENROUTER), http_client=http)
    assert all(isinstance(c, OpenRouterClient) for c in clients.clients.values())
    assert clients.models == DEFAULT_MODELS[Provider.OPENROUTER]
    await clients.aclose()
    assert not http.is_closed  # a passed client is the caller's to close
    await http.aclose()


def test_live_provider_without_a_key_is_refused() -> None:
    with pytest.raises(ModelConfigError, match="GEMINI_API_KEY"):
        build_model_clients(settings(llm_provider=Provider.GEMINI))
    with pytest.raises(ModelConfigError, match="OPENROUTER_API_KEY"):
        build_model_clients(settings(llm_provider=Provider.OPENROUTER))


async def test_record_mode_wraps_live_clients(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    s = settings(llm_provider=Provider.GEMINI, record=True, recordings_dir=tmp_path)
    clients = build_model_clients(s, case_id="happy-001")
    assert all(isinstance(c, RecordingClient) for c in clients.clients.values())
    await clients.aclose()
    with pytest.raises(ModelConfigError, match="needs a case id"):
        build_model_clients(s)


def test_requests_per_minute_defaults_to_the_free_tier_quota() -> None:
    assert Settings(_env_file=None, llm_provider="gemini").requests_per_minute() == 15
    assert Settings(_env_file=None, llm_provider="openrouter").requests_per_minute() == 20
    assert Settings(_env_file=None).requests_per_minute() is None  # replay
    tuned = Settings(_env_file=None, llm_provider="gemini", llm_requests_per_minute=5)
    assert tuned.requests_per_minute() == 5
    off = Settings(_env_file=None, llm_provider="gemini", llm_requests_per_minute=0)
    assert off.requests_per_minute() is None


def test_each_key_draws_on_its_own_quota() -> None:
    # Two Gemini projects (keys) on one model have separate quotas, so separate windows.
    def pacer(key: str) -> object:
        s = Settings(_env_file=None, llm_provider="gemini", gemini_api_key=key)
        return build_model_clients(s).clients["triage"]._limiter  # type: ignore[attr-defined]

    assert pacer("a" * 40) is pacer("a" * 40)
    assert pacer("a" * 40) is not pacer("b" * 40)
    one = Settings(_env_file=None, llm_provider="gemini", gemini_api_key="a" * 40)
    assert "a" * 40 not in one.quota_id()


def test_clients_on_one_model_share_a_pacer() -> None:
    settings = Settings(_env_file=None, llm_provider="gemini", gemini_api_key="k" * 20)
    first = build_model_clients(settings)  # the process-wide default
    second = build_model_clients(settings, limits=RateLimits.shared(15, quota=settings.quota_id()))
    pacers = {
        id(c._limiter)  # type: ignore[attr-defined]
        for clients in (first, second)
        for role, c in clients.clients.items()
        if clients.models[role] == "gemini-3.5-flash-lite"
    }
    assert len(pacers) == 1
