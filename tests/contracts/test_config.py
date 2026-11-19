from __future__ import annotations

import pytest

from triagewright import api, cli, mcp_server
from triagewright.config import Settings
from triagewright.contracts.llm import Provider
from triagewright.evals import main as evals_main


def test_defaults_are_safe_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "TW_LLM_PROVIDER"):
        monkeypatch.delenv(var, raising=False)
    s = Settings(_env_file=None)
    assert s.llm_provider is Provider.REPLAY
    assert s.anthropic_api_key is None and s.openrouter_api_key is None
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


def test_evals_smoke_validates_cases(capsys: pytest.CaptureFixture[str]) -> None:
    assert evals_main(["smoke"]) == 0
    assert "validated" in capsys.readouterr().out
    assert evals_main([]) == 2


@pytest.mark.parametrize("entry", [api.main, cli.main, mcp_server.main])
def test_unimplemented_entry_points_exit_cleanly(entry: object) -> None:
    with pytest.raises(SystemExit, match="not implemented"):
        entry()  # type: ignore[operator]
