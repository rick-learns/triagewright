from __future__ import annotations

import pytest

from triagewright.config import DEFAULT_MODELS, Settings, is_free_model
from triagewright.contracts.llm import Provider
from triagewright.evals import main as evals_main


def test_defaults_are_safe_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("OPENROUTER_API_KEY", "GEMINI_API_KEY", "TW_LLM_PROVIDER", "TW_APPROVAL_KEY"):
        monkeypatch.delenv(var, raising=False)
    s = Settings(_env_file=None)
    assert s.llm_provider is Provider.REPLAY
    assert s.openrouter_api_key is None and s.gemini_api_key is None and s.approval_key is None
    assert s.models() == DEFAULT_MODELS[Provider.GEMINI]  # recordings are made with Gemini
    assert s.loop_config().max_turns == 8
    assert s.policy_config().refund_hard_limit_cents == 50_000
    assert set(s.models()) == {"coordinator", "triage", "answer", "action", "escalation", "judge"}


def test_env_overrides_and_secrets_are_masked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TW_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("TW_MAX_TURNS", "3")
    s = Settings(_env_file=None)
    assert s.llm_provider is Provider.OPENROUTER and s.max_turns == 3
    assert s.openrouter_api_key is not None
    assert "sk-test-not-real" not in repr(s)


def test_blank_secrets_are_unset_and_paid_models_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TW_APPROVAL_KEY", "  ")
    monkeypatch.setenv("TW_APPROVER_TOKEN", "")
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    s = Settings(_env_file=None)
    assert s.approval_key is None and s.approver_token is None and s.openrouter_api_key is None
    monkeypatch.setenv("TW_MCP_TRANSPORT", "http")
    with pytest.raises(ValueError, match="TW_APPROVAL_KEY is required"):
        Settings(_env_file=None)
    monkeypatch.setenv("TW_APPROVAL_KEY", "short")
    with pytest.raises(ValueError, match="at least 32"):
        Settings(_env_file=None)
    monkeypatch.setenv("TW_APPROVAL_KEY", "k" * 32)
    assert Settings(_env_file=None).approval_key is not None
    monkeypatch.setenv("TW_MODEL_ANSWER", "anthropic/claude-haiku-5-5")
    with pytest.raises(ValueError, match="only free replay models"):
        Settings(_env_file=None)


@pytest.mark.parametrize(
    ("provider", "model", "free"),
    [
        (Provider.GEMINI, "gemini-3.5-flash-lite", True),
        (Provider.GEMINI, "gemini-3.8-flash", True),
        (Provider.GEMINI, "gemma-4-31b-it", True),
        (Provider.GEMINI, "gemma-4-31b-it-latest", False),
        (Provider.GEMINI, "gemini-3-flash-preview", True),
        (Provider.GEMINI, "gemini-3.5-pro", False),
        (Provider.GEMINI, "gemini-flash-latest", False),
        (Provider.GEMINI, "gemini-4-argon", False),
        (Provider.GEMINI, "google/gemma-4-31b-it:free", False),
        (Provider.OPENROUTER, "google/gemma-4-31b-it:free", True),
        (Provider.OPENROUTER, "gemini-3.5-flash-lite", False),
        (Provider.REPLAY, "gemini-3.5-flash-lite", True),
        (Provider.REPLAY, "anthropic/claude-haiku-5-5", False),
    ],
)
def test_free_models_per_provider(provider: Provider, model: str, free: bool) -> None:
    assert is_free_model(provider, model) is free


def test_each_live_provider_has_free_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TW_MODEL_JUDGE", "")  # blank override means the default
    for provider in (Provider.GEMINI, Provider.OPENROUTER):
        monkeypatch.setenv("TW_LLM_PROVIDER", provider.value)
        s = Settings(_env_file=None)
        assert s.models() == DEFAULT_MODELS[provider]
    monkeypatch.setenv("TW_MODEL_ANSWER", "gemma-4-31b-it")
    with pytest.raises(ValueError, match="only free openrouter models"):
        Settings(_env_file=None)


def test_example_env_loads_with_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    from triagewright.testing import REPO_ROOT

    for var in ("OPENROUTER_API_KEY", "GEMINI_API_KEY", "TW_APPROVAL_KEY", "TW_APPROVER_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    s = Settings(_env_file=REPO_ROOT / ".env.example")
    assert s.approval_key is None and s.approver_token is None


def test_evals_smoke_validates_cases(capsys: pytest.CaptureFixture[str]) -> None:
    assert evals_main(["smoke"]) == 0
    assert "validated" in capsys.readouterr().out
    assert evals_main([]) == 2
