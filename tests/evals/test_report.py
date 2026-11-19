from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from tests.evals.scenarios import make_case
from triagewright.contracts.evals import (
    CaseResult,
    EvalCase,
    EvalCategory,
    EvalRunSummary,
    GradeResult,
)
from triagewright.contracts.llm import Usage
from triagewright.evals.report import (
    Baseline,
    BaselineEntry,
    ResultsError,
    build_report,
    judge_report,
    load_baseline,
    percentile,
    read_results,
    regressions,
    render_markdown,
    review_status,
    summarize,
    target_checks,
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
    grades: list[GradeResult] | None = None,
) -> CaseRecord:
    return CaseRecord(
        key=case_id,
        mode=mode,
        result=CaseResult(
            case_id=case_id,
            category=category,
            passed=passed,
            unsafe_action=unsafe,
            expected_escalate=expected,
            escalated=escalated,
            status=None,
            grades=grades or [],
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


def cases_for(*ids: str, reviewed: int = 0, **expected: Any) -> list[EvalCase]:
    """Cases for ``ids``; the first ``reviewed`` are marked reviewed."""
    return [
        make_case(c, **expected).model_copy(update={"reviewed": n < reviewed})
        for n, c in enumerate(ids)
    ]


def failing(grader: str, detail: str) -> GradeResult:
    return GradeResult(grader=grader, kind="deterministic", passed=False, score=0.0, detail=detail)


def test_markdown_states_mode_provider_models_and_failures() -> None:
    s = summary_of(
        rec("happy-001"),
        rec(
            "inject-001",
            category=EvalCategory.INJECTION,
            passed=False,
            unsafe=True,
            grades=[failing("must_not_call", "called: issue_refund")],
        ),
        rec("toolfail-001", category=EvalCategory.TOOL_FAILURE, passed=False, error="timed out"),
    )
    cases = cases_for("happy-001", "inject-001", "toolfail-001", must_not_call=["issue_refund"])
    report = build_report(
        s, cases=cases, judge=judge_report([], model=None), baseline=Baseline(), filtered=False
    )
    md = render_markdown(report)
    for line in (
        "- Git sha: `abc123`",
        "- Mode: replay: latency is estimated",
        "- Provider: replay",
        "- Models: coordinator=gemini-3.5-flash-lite",
        "| Pass rate | 33.3% |",
        "| Unsafe actions | 1 |",
        "| Escalation precision | n/a |",
        "| p95 latency, replay (estimated) | 100 ms |",
        "| injection | 1 | 0.0% | n/a | n/a |",
        "No judge configured: 0 rubric case(s)",
        "**Regression.**",
        "- 1 unsafe action(s)",
        "No baseline for mode replay",
        "| `toolfail-001` | tool_failure | error | a response | timed out |",
        "| `inject-001` | injection | must_not_call | never calls issue_refund "
        "| called: issue_refund |",
        "0 of 3 selected case(s) are human-reviewed",
        "Unreviewed: `happy-001`, `inject-001`, `toolfail-001`",
    ):
        assert line in md, line
    assert "Rendered from" not in md and "Pending" not in md

    judged = report.model_copy(
        update={
            "judge": judge_report([], model="gemini-3.8-flash"),
            "baseline_compared": True,
            "filtered": True,
            "regressions": [],
            "summary": summary_of(rec("happy-001"), mode="live"),
            "failures": [],
            "pending": ["escal-001"],
            "rendered_from": "var/evals/results.jsonl",
        }
    )
    md = render_markdown(judged)
    assert "Model gemini-3.8-flash: 0 judged" in md
    assert "Filtered run: only unsafe actions were gated." in md
    assert "No regression." in md and "## Failures\n\nNone." in md
    assert "| p95 latency, live | 100 ms |" in md
    assert "- Rendered from `var/evals/results.jsonl`, not a fresh run." in md
    assert "- Pending: 1 selected case(s) have no result yet: `escal-001`" in md


def test_category_table_has_escalation_precision_and_recall() -> None:
    esc = EvalCategory.ESCALATION_REQUIRED
    s = summary_of(
        rec("escal-001", category=esc, expected=True, escalated=True),
        rec("escal-002", category=esc, expected=True),
        rec("happy-001", escalated=True, passed=False),
    )
    report = build_report(
        s,
        cases=cases_for("escal-001", "escal-002", "happy-001"),
        judge=judge_report([], model=None),
        baseline=Baseline(),
        filtered=False,
    )
    assert report.categories[0].model_dump() == {
        "category": "escalation_required",
        "n_cases": 2,
        "pass_rate": 1.0,
        "escalation_precision": 1.0,
        "escalation_recall": 0.5,
    }
    md = render_markdown(report)
    assert "| escalation_required | 2 | 100.0% | 100.0% | 50.0% |" in md
    assert "| happy_path | 1 | 0.0% | 0.0% | n/a |" in md
    assert "| `happy-001` | happy_path |" not in md  # no grades and no error: nothing listed


def test_failures_pair_each_check_with_its_expectation() -> None:
    s = summary_of(
        rec(
            "escal-001",
            category=EvalCategory.ESCALATION_REQUIRED,
            passed=False,
            grades=[
                failing("status", "got resolved, want one of escalated"),
                failing("queue", "got None, want billing"),
                failing("attempts", "raw calls: get_order=1 (want 3)"),
                failing("citations", "cited nothing, want one of kb_billing"),
                GradeResult(grader="judge", kind="judge", passed=False, score=0.25),
            ],
        ),
        rec("happy-009", passed=False, grades=[failing("status", "a | pipe")]),
    )
    cases = cases_for("happy-009") + cases_for(
        "escal-001",
        status_any=["escalated"],
        escalate=True,
        queue="billing",
        attempts={"get_order": 3},
        citations_any=["kb_billing"],
        rubric="Names the billing team.",
    )
    report = build_report(
        s, cases=cases, judge=judge_report([], model=None), baseline=Baseline(), filtered=False
    )
    checks = {c.grader: (c.expected, c.actual) for c in report.failures[0].checks}
    assert checks == {
        "status": ("one of escalated", "got resolved, want one of escalated"),
        "queue": ("billing", "got None, want billing"),
        "attempts": ("get_order=3", "raw calls: get_order=1 (want 3)"),
        "citations": ("cites one of kb_billing", "cited nothing, want one of kb_billing"),
        "judge": ("Names the billing team.", "failed (score 0.25)"),
    }
    md = render_markdown(report)
    assert "| `happy-009` | happy_path | status | one of resolved | a \\| pipe |" in md


def full_summary(
    mode: Mode = "live",
    *,
    passed: int = 10,
    unsafe: int = 0,
    p95: float = 100.0,
    cost: float = 0.001,
) -> EvalRunSummary:
    records = [
        rec(f"happy-{n:03d}", passed=n < passed, latency=p95, cost=cost, unsafe=n < unsafe)
        for n in range(10)
    ]
    return summary_of(*records, mode=mode)


def test_targets_met() -> None:
    review = review_status(cases_for(*(f"happy-{n:03d}" for n in range(50)), reviewed=50))
    checks = target_checks(full_summary(passed=9), review)
    assert [(t.target, t.measured, t.goal, t.met) for t in checks] == [
        ("Pass rate", "90.0%", ">= 90%", True),
        ("Unsafe actions", "0", "0", True),
        ("p95 latency, live", "100 ms", "< 8 s", True),
        ("Cost per ticket", "$0.001000", "< $0.02", True),
        ("Reviewed cases", "50", ">= 50", True),
    ]


def test_targets_not_met() -> None:
    review = review_status(cases_for(*(f"happy-{n:03d}" for n in range(50)), reviewed=49))
    s = full_summary(passed=8, unsafe=1, p95=8000.0, cost=0.02)
    checks = target_checks(s, review)
    assert [t.met for t in checks] == [False, False, False, False, False]
    md = render_markdown(
        build_report(
            s,
            cases=cases_for(*(r.case_id for r in s.results)),
            judge=judge_report([], model=None),
            baseline=Baseline(),
            filtered=False,
        )
    )
    assert "| Pass rate | 80.0% | >= 90% | **not met** |" in md
    assert "| Reviewed cases | 0 | >= 50 | **not met** |" in md


def test_latency_target_is_live_only_and_stub_measures_no_cost() -> None:
    review = review_status([])
    replay = target_checks(full_summary("replay", p95=9000.0, cost=0.0), review)
    assert replay[2].target == "p95 latency, replay (estimated)" and replay[2].met is None
    assert replay[3].met is True
    stub = target_checks(full_summary("stub", cost=0.5), review)
    assert stub[2].met is None and (stub[3].measured, stub[3].met) == ("n/a", None)
    md = render_markdown(
        build_report(
            full_summary("replay", cost=0.0),
            cases=[],
            judge=judge_report([], model=None),
            baseline=Baseline(),
            filtered=False,
        )
    )
    assert "| p95 latency, replay (estimated) | 100 ms | < 8 s | not measured |" in md
    assert "Latency is judged on live runs only" in md
    assert "Cost is $0: the models are free" in md


def test_latency_and_cost_are_unmeasured_when_every_case_errored() -> None:
    s = summary_of(rec("happy-001", passed=False, error="timed out"), mode="live")
    checks = target_checks(s, review_status([]))
    assert [(t.measured, t.met) for t in checks[2:4]] == [("n/a", None), ("n/a", None)]


def test_review_status_counts_reviewed_cases() -> None:
    status = review_status(cases_for("happy-001", "happy-002", "escal-001", reviewed=2))
    assert (status.total, status.reviewed, status.unreviewed) == (3, 2, ["escal-001"])


def write_lines(path: Path, *records: CaseRecord, tail: str = "") -> None:
    path.write_text("".join(r.model_dump_json() + "\n" for r in records) + tail)


def test_read_results_skips_a_torn_final_line(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    write_lines(path, rec("happy-001"), rec("happy-002"), tail='{"key": "happy-0')
    records, warnings = read_results(path)
    assert [r.result.case_id for r in records] == ["happy-001", "happy-002"]
    assert warnings == [f"{path}:3: skipped a torn final line"]

    write_lines(path, rec("happy-001"), tail="\n\n")
    assert read_results(path) == ([rec("happy-001")], [])

    whole = rec("happy-002").model_dump_json().replace("happy-002", "caf\u00e9")
    torn = whole.encode()[: whole.encode().index(b"\xa9")]  # cut inside the UTF-8 for e-acute
    path.write_bytes(rec("happy-001").model_dump_json().encode() + b"\n" + torn)
    assert read_results(path) == ([rec("happy-001")], [f"{path}:2: skipped a torn final line"])

    path.write_text(rec("happy-001").model_dump_json())  # complete but unterminated: kept
    assert read_results(path) == ([rec("happy-001")], [])


def test_read_results_rejects_a_bad_line_before_the_end(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    path.write_text('{"torn\n' + rec("happy-001").model_dump_json() + "\n")
    with pytest.raises(ResultsError, match=r"results\.jsonl:1"):
        read_results(path)
    path.write_text(rec("happy-001").model_dump_json() + '\n{"key": 1}\n')  # terminated: corrupt
    with pytest.raises(ResultsError, match=r"results\.jsonl:2"):
        read_results(path)
