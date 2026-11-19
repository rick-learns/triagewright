"""The built report and its markdown: header, categories, and failures."""

from __future__ import annotations

from tests.evals.test_report_support import cases_for, failing, rec, summary_of
from triagewright.contracts.evals import EvalCategory, GradeResult
from triagewright.evals.report import Baseline, build_report, judge_report, render_markdown


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
        "0 of 3 selected case(s) are reviewed (`reviewed: true`; see evals/REVIEWING.md).",
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


def test_zero_cost_does_not_claim_the_models_are_free() -> None:
    """$0 can mean free models or no billable usage; the results don't record which."""
    report = build_report(
        summary_of(rec("happy-001", cost=0.0)),
        cases=cases_for("happy-001"),
        judge=judge_report([], model=None),
        baseline=Baseline(),
        filtered=False,
    )
    md = render_markdown(report)
    assert (
        "Cost is $0: the models are free, or the tickets it counts (errored cases and the "
        "judge are excluded) recorded no billable usage, so the cost target is met trivially." in md
    )
