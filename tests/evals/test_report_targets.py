"""The release targets and the review count."""

from __future__ import annotations

from tests.evals.test_report_support import cases_for, rec, summary_of
from triagewright.contracts.evals import EvalRunSummary
from triagewright.evals.report import (
    Baseline,
    build_report,
    judge_report,
    render_markdown,
    review_status,
    target_checks,
)
from triagewright.evals.runner import Mode


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
    report = build_report(
        s,
        cases=cases_for("happy-001"),
        judge=judge_report([], model=None),
        baseline=Baseline(),
        filtered=False,
    )
    md = render_markdown(report)
    assert "| Cost per ticket (judge excluded) | n/a (no responses) |" in md
    assert "| p95 latency, live | n/a (no responses) |" in md
    assert "Cost is $0" not in md


def test_review_status_counts_reviewed_cases() -> None:
    status = review_status(cases_for("happy-001", "happy-002", "escal-001", reviewed=2))
    assert (status.total, status.reviewed, status.unreviewed) == (3, 2, ["escal-001"])
