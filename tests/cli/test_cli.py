from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import SecretStr
from typer.testing import CliRunner

import triagewright.cli as cli
from tests.api.stack import TOKEN, Stack, fake_stack
from triagewright.api.services import Services, WiringUnavailableError
from triagewright.config import Settings
from triagewright.contracts.agents import ActionOutcome, ActionStatus, FinalResponse
from triagewright.contracts.guardrails import ApprovalStatus
from triagewright.testing import sample_ticket, seed_as_of, seed_world

runner = CliRunner()


@pytest.fixture
def stack() -> Stack:
    return fake_stack()


def invoke(
    stack: Stack,
    *args: str,
    configured: str | None = TOKEN,
    presented: str | None = TOKEN,
) -> tuple[int, str, str]:
    app = cli.create_cli(
        services_loader=lambda settings: stack.services,
        settings_loader=lambda: Settings(
            _env_file=None, approver_token=SecretStr(configured) if configured else None
        ),
        clock=seed_as_of,
    )
    env = {cli.APPROVER_TOKEN_ENV: presented}
    result = runner.invoke(app, list(args), env=env)
    return result.exit_code, result.stdout, result.stderr


def write_ticket(tmp_path: Path, suffix: str) -> Path:
    data = sample_ticket().model_dump(mode="json")
    path = tmp_path / f"ticket{suffix}"
    path.write_text(json.dumps(data) if suffix == ".json" else yaml.safe_dump(data))
    return path


@pytest.mark.parametrize("suffix", [".json", ".yaml", ".yml"])
def test_resolve_a_ticket_file(stack: Stack, tmp_path: Path, suffix: str) -> None:
    code, out, _ = invoke(stack, "resolve", str(write_ticket(tmp_path, suffix)))
    assert code == 0
    assert FinalResponse.model_validate_json(out).ticket_id == "tkt_0001"


def test_resolve_from_flags(stack: Stack) -> None:
    code, out, _ = invoke(
        stack, "resolve", "--ticket-id", "tkt_0002", "--customer-id", "cus_ava01", "--body", "hi"
    )
    assert code == 0
    assert FinalResponse.model_validate_json(out).ticket_id == "tkt_0002"


def test_resolve_usage_errors(stack: Stack, tmp_path: Path) -> None:
    ticket = str(write_ticket(tmp_path, ".json"))
    assert invoke(stack, "resolve")[0] == cli.ExitCode.USAGE
    assert invoke(stack, "resolve", ticket, "--body", "x")[0] == cli.ExitCode.USAGE
    bad = tmp_path / "ticket.txt"
    bad.write_text("{}")
    assert invoke(stack, "resolve", str(bad))[0] == cli.ExitCode.USAGE
    broken = tmp_path / "broken.json"
    broken.write_text("{nope")
    code, _, err = invoke(stack, "resolve", str(broken))
    assert code == cli.ExitCode.USAGE and "not valid JSON" in err
    code, _, err = invoke(stack, "resolve", "--ticket-id", "card-4111111111111111", "--body", "x")
    assert code == cli.ExitCode.USAGE and "ticket_id" in err
    assert "4111111111111111" not in err


def test_approvals_list(stack: Stack) -> None:
    code, out, _ = invoke(stack, "approvals", "list")
    assert code == 0
    assert [a["approval_id"] for a in json.loads(out)] == [stack.approval_id]


def test_approve_executes(stack: Stack) -> None:
    code, out, _ = invoke(stack, "approvals", "decide", stack.approval_id, "--approve")
    assert code == 0
    assert ActionOutcome.model_validate_json(out).status is ActionStatus.EXECUTED
    assert len(stack.gateway.refunds) == len(seed_world().refunds) + 1
    code, _, err = invoke(stack, "approvals", "decide", stack.approval_id, "--approve")
    assert code == cli.ExitCode.CONFLICT and "already executed" in err


def test_reject_then_conflict(stack: Stack) -> None:
    code, out, _ = invoke(stack, "approvals", "decide", stack.approval_id, "--reject")
    assert code == 0 and ActionOutcome.model_validate_json(out).status is ActionStatus.SKIPPED
    code, _, _ = invoke(stack, "approvals", "decide", stack.approval_id, "--approve")
    assert code == cli.ExitCode.CONFLICT
    assert stack.gateway.refunds == seed_world().refunds


def test_failed_execution_exits_1(stack: Stack) -> None:
    stack.gateway.orders.pop("ord_h8001")
    code, out, _ = invoke(stack, "approvals", "decide", stack.approval_id, "--approve")
    assert code == cli.ExitCode.FAILED
    assert ActionOutcome.model_validate_json(out).status is ActionStatus.FAILED


@pytest.mark.parametrize(
    ("configured", "presented", "expected"),
    [
        (None, TOKEN, cli.ExitCode.UNAVAILABLE),
        (TOKEN, None, cli.ExitCode.UNAUTHORIZED),
        (TOKEN, "wrong-token", cli.ExitCode.UNAUTHORIZED),
    ],
)
def test_decide_token_matrix(
    stack: Stack, configured: str | None, presented: str | None, expected: int
) -> None:
    code, out, err = invoke(
        stack,
        "approvals",
        "decide",
        stack.approval_id,
        "--approve",
        configured=configured,
        presented=presented,
    )
    assert code == expected and out == ""
    assert TOKEN not in err
    stored = stack.approvals.get(stack.approval_id)
    assert stored is not None and stored.status is ApprovalStatus.PENDING


def test_decide_unknown_approval(stack: Stack) -> None:
    code, _, _ = invoke(stack, "approvals", "decide", "apr_000000000000", "--approve")
    assert code == cli.ExitCode.NOT_FOUND


def test_decide_requires_a_decision(stack: Stack) -> None:
    code, _, _ = invoke(stack, "approvals", "decide", stack.approval_id)
    assert code == cli.ExitCode.USAGE


def test_unwired_cli_explains_itself() -> None:
    def unwired(settings: Settings) -> Services:
        raise WiringUnavailableError("the agent isn't wired yet (task X1)")

    app = cli.create_cli(services_loader=unwired, settings_loader=lambda: Settings(_env_file=None))
    result = runner.invoke(app, ["approvals", "list"])
    assert result.exit_code == cli.ExitCode.UNAVAILABLE and "task X1" in result.stderr


def test_main_runs_the_typer_app(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["triagewright", "--help"])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 0
