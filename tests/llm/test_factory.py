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
