"""The JSON report (:class:`EvalReport`) and :func:`build_report`, which assembles it.

``pending`` and ``rendered_from`` cover ``triagewright-evals report`` re-rendering a results
file mid-run.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from triagewright.contracts.common import Contract
from triagewright.contracts.evals import EvalCase, EvalRunSummary
from triagewright.evals.report.aggregate import CategoryStats, JudgeReport, category_stats
from triagewright.evals.report.baseline import Baseline, regressions
from triagewright.evals.report.failures import FailureDetail, failure_details
from triagewright.evals.report.targets import (
    ReviewStatus,
    TargetCheck,
    review_status,
    target_checks,
)


class EvalReport(Contract):
    """The JSON report: the contract summary plus judge usage and the baseline verdict."""

    summary: EvalRunSummary
    judge: JudgeReport
    baseline_compared: bool  # False when the baseline has no entry for this mode
    filtered: bool  # a case filter was used, so only unsafe actions were gated
    regressions: list[str]
    categories: list[CategoryStats]
    review: ReviewStatus
    targets: list[TargetCheck]
    failures: list[FailureDetail]
    pending: list[str] = []  # selected cases with no result yet (rendered mid-run)
    rendered_from: str | None = None  # the results JSONL, for ``triagewright-evals report``
    # "run": summary.git_sha is the checkout that ran; "render": the one that rendered the
    # results, which don't record the sha they ran at.
    git_sha_source: Literal["run", "render"] = "run"


def build_report(
    summary: EvalRunSummary,
    *,
    cases: Sequence[EvalCase],
    judge: JudgeReport,
    baseline: Baseline,
    filtered: bool,
    pending: Sequence[str] = (),
    rendered_from: str | None = None,
) -> EvalReport:
    """The report for ``summary``. ``cases`` are the selected cases, pending ones included."""
    review = review_status(cases)
    return EvalReport(
        summary=summary,
        judge=judge,
        baseline_compared=summary.mode in baseline.modes,
        filtered=filtered,
        regressions=regressions(summary, baseline, compare_metrics=not filtered),
        categories=category_stats(summary),
        review=review,
        targets=target_checks(summary, review),
        failures=failure_details(summary.results, {c.id: c for c in cases}),
        pending=list(pending),
        rendered_from=rendered_from,
        git_sha_source="render" if rendered_from else "run",
    )
