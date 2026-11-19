from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from triagewright.contracts.evals import CaseResult, EvalCategory, EvalRunSummary
from triagewright.contracts.llm import Usage
from triagewright.evals.report import (
    Baseline,
    BaselineEntry,
    EvalReport,
    judge_report,
    load_baseline,
    percentile,
    regressions,
    render_markdown,
    summarize,
    updated_baseline,
)
from triagewright.evals.runner import CaseRecord, JudgeStatus, Mode
from triagewright.testing import REPO_ROOT

NOW = datetime(2026, 11, 19, tzinfo=UTC)


def rec(
    case_id: str,
    *,
    passed: bool = True,
    expected: bool = False,
    escalated: bool = False,
    latency: float = 100.0,
    cost: float = 0.0,
    unsafe: bool = False,
    error: str | None = None,
    category: EvalCategory = EvalCategory.HAPPY_PATH,
    mode: Mode = "replay",
    judge_status: JudgeStatus = "none",
    judge_usage: Usage | None = None,
) -> CaseRecord:
    return CaseRecord(
        mode=mode,
        result=CaseResult(
            case_id=case_id,
            category=category,
            passed=passed,
            unsafe_action=unsafe,
            expected_escalate=expected,
            escalated=escalated,
            status=None,
            grades=[],
            cost_usd=cost,
            latency_ms=latency,
            error=error,
        ),
        judge_status=judge_status,
        judge_usage=judge_usage,
    )


def summary_of(*records: CaseRecord, mode: Mode = "replay") -> EvalRunSummary:
    return summarize(
        records,
        mode=mode,
        provider="replay",
        models={"coordinator": "gemini-3.5-flash-lite"},
        git_sha="abc123",
        created_at=NOW,
    )


@pytest.mark.parametrize(
    ("values", "q", "want"),
    [
        ([], 95, 0.0),
        ([7.0], 95, 7.0),
        ([1.0, 2.0, 3.0, 4.0], 50, 2.5),
        ([4.0, 1.0, 3.0, 2.0], 95, 3.85),  # unsorted input; 1 + 0.95 * 3 = 3.85 -> 3.85
        ([float(i) for i in range(1, 21)], 95, 19.05),
        ([10.0, 20.0], 0, 10.0),
        ([10.0, 20.0], 100, 20.0),
    ],
)
def test_percentile_interpolates_linearly(values: list[float], q: float, want: float) -> None:
    assert percentile(values, q) == pytest.approx(want)


def test_summary_math() -> None:
    s = summary_of(
        # TP, FP, FN, TN: precision 1/2, recall 1/2
        rec("escal-001", category=EvalCategory.ESCALATION_REQUIRED, expected=True, escalated=True),
        rec("escal-002", category=EvalCategory.ESCALATION_REQUIRED, passed=False, escalated=True),
        rec("happy-001", passed=False, expected=True, latency=300.0, cost=0.003),
        rec("happy-002", latency=200.0, cost=0.001),
        rec("happy-003", passed=False, error="timed out", latency=60_000.0, cost=9.0),
    )
    assert s.n_cases == 5
    assert s.pass_rate == pytest.approx(2 / 5)
    assert s.escalation_precision == pytest.approx(0.5)
    assert s.escalation_recall == pytest.approx(0.5)
    assert s.by_category == {"escalation_required": 0.5, "happy_path": pytest.approx(1 / 3)}
    # The errored case is excluded from cost and latency: [100, 100, 300, 200].
    assert s.cost_per_ticket_usd == pytest.approx(0.004 / 4)
    assert s.p50_latency_ms == pytest.approx(150.0)
    assert s.p95_latency_ms == pytest.approx(285.0)
    assert (s.mode, s.provider, s.git_sha, s.created_at) == ("replay", "replay", "abc123", NOW)
    assert s.run_id.startswith("run_") and len(s.results) == 5


def test_summary_unsafe_and_undefined_rates() -> None:
    s = summary_of(rec("inject-001", unsafe=True, passed=False), rec("inject-002"))
    assert s.unsafe_actions == 1
    assert s.escalation_precision is None and s.escalation_recall is None
    empty = summary_of()
    assert (empty.n_cases, empty.pass_rate, empty.p95_latency_ms) == (0, 0.0, 0.0)


def entry(pass_rate: float = 0.8, cost: float = 0.01, p95: float = 1000.0) -> BaselineEntry:
    return BaselineEntry(
        pass_rate=pass_rate,
        cost_per_ticket_usd=cost,
        p95_latency_ms=p95,
        n_cases=10,
        git_sha="old",
        provider="replay",
        models={},
        created_at=NOW,
    )


def with_metrics(pass_rate: float, cost: float, p95: float, *, unsafe: int = 0) -> EvalRunSummary:
    s = summary_of(rec("happy-001"))
    update = {"pass_rate": pass_rate, "cost_per_ticket_usd": cost, "p95_latency_ms": p95}
    return s.model_copy(update={**update, "unsafe_actions": unsafe})


def test_no_regression_within_tolerance() -> None:
    base = Baseline(modes={"replay": entry()})
    assert regressions(with_metrics(0.8, 0.011, 1100.0), base) == []  # exactly +10% is allowed
    assert regressions(with_metrics(0.9, 0.0, 10.0), base) == []


def test_each_metric_regresses() -> None:
    base = Baseline(modes={"replay": entry()})
    found = regressions(with_metrics(0.7, 0.0111, 1101.0, unsafe=2), base)
    assert len(found) == 4
    assert found[0] == "2 unsafe action(s)"
    assert found[1].startswith("pass rate dropped: 70.0% < 80.0%")
    assert found[2].startswith("cost per ticket rose more than 10%")
    assert found[3].startswith("p95 latency rose more than 10%")


def test_free_baseline_cost_regresses_on_any_spend() -> None:
    base = Baseline(modes={"replay": entry(cost=0.0)})
    assert regressions(with_metrics(0.8, 0.0, 1000.0), base) == []
    assert len(regressions(with_metrics(0.8, 0.0001, 1000.0), base)) == 1


def test_baseline_is_per_mode() -> None:
    base = Baseline(modes={"live": entry(pass_rate=1.0, p95=1.0)})
    assert regressions(with_metrics(0.1, 1.0, 9e9), base) == []  # no replay baseline
    assert regressions(with_metrics(0.1, 1.0, 9e9, unsafe=1), base) == ["1 unsafe action(s)"]


def test_filtered_runs_gate_only_unsafe_actions() -> None:
    base = Baseline(modes={"replay": entry()})
    assert regressions(with_metrics(0.0, 1.0, 9e9), base, compare_metrics=False) == []
    found = regressions(with_metrics(0.0, 1.0, 9e9, unsafe=1), base, compare_metrics=False)
    assert found == ["1 unsafe action(s)"]


def test_stub_latency_is_not_gated() -> None:
    base = Baseline(modes={"stub": entry(p95=0.01)})
    stub = summary_of(rec("happy-001", latency=5.0), mode="stub")
    assert regressions(stub, base) == []


def test_baseline_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    assert load_baseline(path) == Baseline()
    s = summary_of(rec("happy-001", latency=50.0, cost=0.002))
    new = updated_baseline(Baseline(modes={"live": entry()}), s)
    path.write_text(new.model_dump_json(indent=2))
    loaded = load_baseline(path)
    assert set(loaded.modes) == {"live", "replay"}
    assert loaded.modes["replay"].p95_latency_ms == 50.0
    assert loaded.modes["replay"].cost_per_ticket_usd == 0.002
    assert regressions(s, loaded) == []


def test_committed_baseline_is_valid() -> None:
    load_baseline(REPO_ROOT / "evals" / "baseline.json")


def test_judge_report_counts_and_tokens() -> None:
    usage = Usage(input_tokens=10, output_tokens=2)
    records = [
        rec("happy-001", judge_status="judged", judge_usage=usage),
        rec("happy-002", judge_status="judged", judge_usage=usage),
        rec("happy-003", judge_status="after_failure"),
        rec("happy-004"),
    ]
    j = judge_report(records, model="gemini-3.8-flash")
    assert (j.configured, j.judged, j.after_failure, j.skipped) == (True, 2, 1, 0)
    assert (j.input_tokens, j.output_tokens) == (20, 4)
    skipped = judge_report([rec("happy-001", judge_status="skipped")], model=None)
    assert not skipped.configured and skipped.skipped == 1


def test_markdown_states_mode_provider_models_and_failures() -> None:
    s = summary_of(
        rec("happy-001"),
        rec("inject-001", category=EvalCategory.INJECTION, passed=False, unsafe=True),
        rec("toolfail-001", category=EvalCategory.TOOL_FAILURE, passed=False, error="timed out"),
    )
    report = EvalReport(
        summary=s,
        judge=judge_report([], model=None),
        baseline_compared=False,
        filtered=False,
        regressions=regressions(s, Baseline()),
    )
    md = render_markdown(report)
    for line in (
        "- Mode: replay: latency is estimated",
        "- Provider: replay",
        "- Models: coordinator=gemini-3.5-flash-lite",
        "| Pass rate | 33.3% |",
        "| Unsafe actions | 1 |",
        "| Escalation precision | n/a |",
        "| p95 latency, replay (estimated) | 100 ms |",
        "| injection | 0.0% |",
        "No judge configured: 0 rubric case(s)",
        "**Regression.**",
        "- 1 unsafe action(s)",
        "No baseline for mode replay",
        "- `toolfail-001` (tool_failure): timed out",
    ):
        assert line in md, line

    judged = report.model_copy(
        update={
            "judge": judge_report([], model="gemini-3.8-flash"),
            "baseline_compared": True,
            "filtered": True,
            "regressions": [],
            "summary": summary_of(rec("happy-001"), mode="live"),
        }
    )
    md = render_markdown(judged)
    assert "Model gemini-3.8-flash: 0 judged" in md
    assert "Filtered run: only unsafe actions were gated." in md
    assert "No regression." in md and "## Failures\n\nNone." in md
    assert "| p95 latency, live | 100 ms |" in md
