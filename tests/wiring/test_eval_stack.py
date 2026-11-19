"""The eval stack: per-case runtimes, faults, the evals CLI, and the judge client."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.agent.helpers import call, finalize, refund_script, review, triage
from tests.wiring.helpers import REFUND_TICKET, scripted, settings
from triagewright import wiring
from triagewright.contracts.agents import ResolutionStatus
from triagewright.contracts.ports import CaseRuntime, CaseRuntimeFactory
from triagewright.contracts.tools import ErrorCategory, ToolName
from triagewright.evals import cli as evals_cli
from triagewright.evals.runner import load_records
from triagewright.llm import ModelConfigError, ReplayClient
from triagewright.testing import sample_ticket, seed_as_of, seed_world

pytestmark = pytest.mark.anyio


async def test_eval_stack_runs_a_refund_through_the_real_executor(tmp_path: Path) -> None:
    factory = wiring.EvalStackFactory(
        settings(tmp_path), model_factory=scripted(refund_script("ord_a1001", 5000))
    )
    assert isinstance(factory, CaseRuntimeFactory)
    case = factory.build(case_id="policy-001", world=seed_world(), as_of=seed_as_of(), faults=())
    assert isinstance(case, CaseRuntime)

    response = await case.resolver.resolve(REFUND_TICKET)

    assert response.status is ResolutionStatus.PENDING_APPROVAL, response
    assert response.metrics is not None and response.metrics.estimated  # replay mode
    (pending,) = case.pending_approvals()
    assert pending.tool is ToolName.ISSUE_REFUND and pending.arguments["amount_cents"] == 5000
    assert case.ledger().refunds == seed_world().refunds  # approval never completes in-request
    lookups = [c for c in case.tool_calls() if c.tool is ToolName.CHECK_REFUND_ELIGIBILITY]
    assert [c.caller for c in lookups] == ["action"]
    assert not (tmp_path / "world.sqlite").exists()  # evals stay in memory


async def test_eval_cases_are_isolated_from_each_other(tmp_path: Path) -> None:
    factory = wiring.EvalStackFactory(
        settings(tmp_path), model_factory=scripted(refund_script("ord_a1001", 5000))
    )
    first = factory.build(case_id="policy-001", world=seed_world(), as_of=seed_as_of(), faults=())
    await first.resolver.resolve(REFUND_TICKET)
    second = factory.build(case_id="policy-002", world=seed_world(), as_of=seed_as_of(), faults=())
    assert first.pending_approvals() and not second.pending_approvals()
    assert not second.tool_calls()


async def test_eval_stack_applies_case_faults(tmp_path: Path) -> None:
    from triagewright.contracts.evals import ToolFault
    from triagewright.contracts.tools import ToolError

    fault = ToolFault(
        tool=ToolName.GET_ORDER,
        error=ToolError(category=ErrorCategory.TRANSIENT, message="down", retryable=True),
    )
    script = [
        triage(),
        call("get_order", {"order_id": "ord_a1002"}),
        finalize(),
        review(),
    ]
    factory = wiring.EvalStackFactory(settings(tmp_path), model_factory=scripted(script))
    case = factory.build(
        case_id="toolfail-001", world=seed_world(), as_of=seed_as_of(), faults=[fault]
    )
    await case.resolver.resolve(sample_ticket())
    attempts = [c for c in case.tool_calls() if c.tool is ToolName.GET_ORDER]
    assert [c.result.is_error for c in attempts] == [True, False]  # faulted once, then retried


def test_case_runtime_factory_reads_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current = settings(tmp_path)
    monkeypatch.setattr(wiring, "get_settings", lambda: current)
    factory = wiring.case_runtime_factory()
    assert isinstance(factory, wiring.EvalStackFactory) and factory.settings is current


def test_evals_cli_runs_the_default_factory_in_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No recordings: the model call misses, the agent hands off, and the run still reports."""
    empty = settings(tmp_path, recordings_dir=tmp_path / "recordings")
    monkeypatch.setattr(wiring, "get_settings", lambda: empty)
    monkeypatch.setattr(evals_cli, "get_settings", lambda: empty)
    code = evals_cli.main(
        [
            "run",
            "--case",
            "happy-001",
            "--report",
            str(tmp_path / "report.md"),
            "--json",
            str(tmp_path / "report.json"),
            "--results",
            str(tmp_path / "results.jsonl"),
            "--baseline",
            str(tmp_path / "baseline.json"),
        ]
    )
    # One of the 54 cases with no baseline gates only unsafe actions, and a handoff takes none.
    assert code == 0
    assert "1 cases (replay): pass rate 0.0%, 0 unsafe" in capsys.readouterr().out
    assert (tmp_path / "report.md").exists()
    (record,) = load_records(tmp_path / "results.jsonl").values()
    assert record.result.error is None and record.result.escalated  # a replay handoff, no error


def test_judge_client_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    current = settings(tmp_path, recordings_dir=tmp_path)
    monkeypatch.setattr(wiring, "get_settings", lambda: current)
    build = wiring.judge_client_factory()
    client = build(case_id="happy-001", recordings=tmp_path / "happy-001.judge.jsonl")
    assert isinstance(client, ReplayClient) and client.model == current.models()["judge"]
    with pytest.raises(ModelConfigError, match="judge recordings"):
        build(case_id="happy-001", recordings=tmp_path / "happy-001.jsonl")
