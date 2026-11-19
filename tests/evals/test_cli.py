from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.evals.scenarios import make_case
from triagewright.contracts.ports import ModelClient
from triagewright.evals import main
from triagewright.evals.report import Baseline, EvalReport
from triagewright.testing import REPO_ROOT, FakeCaseRuntimeFactory, ScriptedModelClient


@pytest.fixture(autouse=True)
def _repo_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(REPO_ROOT)  # settings use repo-relative paths (data/seed, ...)


def write_cases(root: Path, **expected: Any) -> Path:
    cases_dir = root / "cases"
    for case_id in ("escal-001", "happy-001"):
        status = "escalated" if case_id.startswith("escal") else "resolved"
        case = make_case(case_id, status_any=[status], escalate=status == "escalated", **expected)
        path = cases_dir / case.category.value / f"{case_id}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(case.model_dump(mode="json", exclude_none=True)))
    return cases_dir


def run_args(root: Path, *extra: str) -> list[str]:
    return [
        "run",
        "--stub",
        "--cases",
        str(root / "cases"),
        "--report",
        str(root / "report.md"),
        "--json",
        str(root / "report.json"),
        "--results",
        str(root / "results.jsonl"),
        "--baseline",
        str(root / "baseline.json"),
        *extra,
    ]


def test_smoke_validates_cases(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cases_dir = write_cases(tmp_path)
    assert main(["smoke", str(cases_dir)]) == 0
    assert capsys.readouterr().out == f"validated 2 eval cases in {cases_dir}\n"
    assert main(["smoke"]) == 0
    assert capsys.readouterr().out.startswith("validated ")


def test_usage_errors_exit_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2
    assert main(["bogus"]) == 2
    assert main(["run", "--concurrency", "many"]) == 2
    assert main(["--help"]) == 0
    capsys.readouterr()


def test_stub_run_writes_reports_and_gates_on_the_baseline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_cases(tmp_path)
    # The stub resolver claims an escalation but never creates one, so both cases fail.
    assert main(run_args(tmp_path, "--update-baseline")) == 0
    out = capsys.readouterr().out
    assert "2 cases (stub): pass rate 0.0%, 0 unsafe" in out
    assert "baseline for stub written" in out

    report = EvalReport.model_validate_json((tmp_path / "report.json").read_text())
    s = report.summary
    assert (s.mode, s.n_cases, s.pass_rate, s.models) == ("stub", 2, 0.0, {})
    assert s.escalation_precision is None and s.escalation_recall == 0.0
    assert "escalate: want escalate=True" in (tmp_path / "report.md").read_text()
    assert not report.baseline_compared and not report.judge.configured
    assert "- Mode: stub" in (tmp_path / "report.md").read_text()
    assert len((tmp_path / "results.jsonl").read_text().splitlines()) == 2

    baseline = Baseline.model_validate_json((tmp_path / "baseline.json").read_text())
    assert baseline.modes["stub"].pass_rate == 0.0
    assert main(run_args(tmp_path, "--resume")) == 0  # same result: no regression

    raised = baseline.model_copy(
        update={"modes": {"stub": baseline.modes["stub"].model_copy(update={"pass_rate": 1.0})}}
    )
    (tmp_path / "baseline.json").write_text(raised.model_dump_json())
    assert main(run_args(tmp_path)) == 1
    assert "regression: pass rate dropped" in capsys.readouterr().err


def test_case_filter_gates_only_unsafe_and_never_updates_the_baseline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_cases(tmp_path)
    (tmp_path / "baseline.json").write_text(
        json.dumps(
            {
                "modes": {
                    "stub": {
                        "pass_rate": 1.0,
                        "cost_per_ticket_usd": 0.0,
                        "p95_latency_ms": 0.0,
                        "n_cases": 2,
                        "git_sha": "x",
                        "provider": "none",
                        "models": {},
                        "created_at": "2026-11-19T00:00:00Z",
                    }
                }
            }
        )
    )
    assert main(run_args(tmp_path, "--case", "happy-*", "--update-baseline")) == 0
    captured = capsys.readouterr()
    assert "1 cases (stub): pass rate 0.0%" in captured.out
    assert "baseline not updated" in captured.err
    report = EvalReport.model_validate_json((tmp_path / "report.json").read_text())
    assert report.filtered and report.regressions == []


def test_empty_selection_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write_cases(tmp_path)
    assert main(run_args(tmp_path, "--case", "nope-*")) == 2
    assert main([*run_args(tmp_path), "--cases", str(tmp_path / "missing")]) == 2
    assert "no eval cases selected" in capsys.readouterr().err
    assert not (tmp_path / "report.md").exists()


def test_a_filter_matching_every_case_is_a_full_run(tmp_path: Path) -> None:
    write_cases(tmp_path)
    assert main(run_args(tmp_path, "--case", "*")) == 0
    report = EvalReport.model_validate_json((tmp_path / "report.json").read_text())
    assert not report.filtered


def test_missing_wiring_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "triagewright.wiring", None)  # import raises ImportError
    write_cases(tmp_path)
    args = [a for a in run_args(tmp_path) if a != "--stub"]
    assert main(args) == 2
    assert "cannot load the eval stack" in capsys.readouterr().err
    assert main([*args, "--factory", "tests.evals.test_cli:not_a_factory"]) == 2
    assert "did not return a CaseRuntimeFactory" in capsys.readouterr().err
    assert main([*run_args(tmp_path), "--judge", "tests.evals.test_cli:not_a_factory"]) == 2
    assert "did not return a JudgeClientFactory" in capsys.readouterr().err


def not_a_factory() -> object:
    return 42


def stub_factory() -> FakeCaseRuntimeFactory:
    return FakeCaseRuntimeFactory()


def scripted_judges() -> object:
    def make(*, case_id: str, recordings: Path) -> ModelClient:
        return ScriptedModelClient([])  # raises if called

    return make


def test_factory_and_judge_load_from_module_specs(tmp_path: Path) -> None:
    write_cases(tmp_path, rubric="Polite and correct.")
    args = [a for a in run_args(tmp_path) if a != "--stub"]
    args += ["--factory", "tests.evals.test_cli:stub_factory"]
    args += ["--judge", "tests.evals.test_cli:scripted_judges"]
    # Default settings are replay mode, but the stub resolver answers in stub mode.
    assert main(args) == 0
    report = EvalReport.model_validate_json((tmp_path / "report.json").read_text())
    assert report.summary.mode == "replay" and report.summary.provider == "replay"
    assert report.summary.models["judge"] == "gemini-3.8-flash"
    assert report.judge.configured and report.judge.after_failure == 2
    errors = {r.case_id: r.error for r in report.summary.results}
    assert errors["escal-001"] == "mode mismatch: response is stub, run is replay"
