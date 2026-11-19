"""Aggregation into :class:`EvalRunSummary`, the baseline gate, and the markdown/JSON report.

Cost per ticket and latency cover cases that produced a response (no
``error``); errored cases count against the pass rate instead. Judge calls are
never part of cost per ticket; the report lists them separately.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Annotated

from pydantic import Field, ValidationError

from triagewright.contracts.common import Contract, stable_id
from triagewright.contracts.evals import (
    CaseResult,
    EvalCase,
    EvalRunSummary,
    ExpectedOutcome,
    GradeResult,
)
from triagewright.evals.runner import CaseRecord, Mode

COST_TOLERANCE = 0.10  # cost per ticket may rise at most 10% (same mode)
LATENCY_TOLERANCE = 0.10  # p95 latency may rise at most 10% (same mode)
_EPS = 1e-9

_MODE_NOTES: dict[str, str] = {
    "live": "live: latency is wall time per ticket",
    "replay": "replay: latency is estimated (non-model wall time plus recorded model latency)",
    "stub": "stub: no agent or model calls; latency is harness wall time and not gated",
}


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (``q`` in [0, 100]), like numpy's default. 0 if empty."""
    if not values:
        return 0.0
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def escalation_rates(results: Iterable[CaseResult]) -> tuple[float | None, float | None]:
    """Escalation (precision, recall) from created escalations; None when undefined."""
    rs = list(results)
    true_pos = sum(r.escalated and r.expected_escalate for r in rs)
    return (
        _ratio(true_pos, sum(r.escalated for r in rs)),
        _ratio(true_pos, sum(r.expected_escalate for r in rs)),
    )


def summarize(
    records: Sequence[CaseRecord],
    *,
    mode: Mode,
    provider: str,
    models: dict[str, str],
    git_sha: str,
    created_at: datetime,
) -> EvalRunSummary:
    results = [r.result for r in records]
    n = len(results)
    ok = [r for r in results if r.error is None]
    precision, recall = escalation_rates(results)
    categories: dict[str, list[bool]] = {}
    for r in results:
        categories.setdefault(r.category.value, []).append(r.passed)
    return EvalRunSummary(
        run_id=stable_id("run", created_at.isoformat(), git_sha, mode),
        created_at=created_at,
        git_sha=git_sha,
        mode=mode,
        provider=provider,
        models=models,
        n_cases=n,
        pass_rate=sum(r.passed for r in results) / n if n else 0.0,
        unsafe_actions=sum(r.unsafe_action for r in results),
        escalation_precision=precision,
        escalation_recall=recall,
        cost_per_ticket_usd=sum(r.cost_usd for r in ok) / len(ok) if ok else 0.0,
        p50_latency_ms=percentile([r.latency_ms for r in ok], 50),
        p95_latency_ms=percentile([r.latency_ms for r in ok], 95),
        by_category={c: sum(v) / len(v) for c, v in sorted(categories.items())},
        results=results,
    )


# ---- judge usage ------------------------------------------------------------------


class JudgeReport(Contract):
    configured: bool
    model: str | None = None
    judged: Annotated[int, Field(ge=0)] = 0
    skipped: Annotated[int, Field(ge=0)] = 0  # rubric cases with no judge configured
    after_failure: Annotated[int, Field(ge=0)] = 0  # rubric cases that failed deterministically
    input_tokens: Annotated[int, Field(ge=0)] = 0
    output_tokens: Annotated[int, Field(ge=0)] = 0
    cost_usd: Annotated[float, Field(ge=0)] = 0.0  # provider-reported only


def judge_report(records: Sequence[CaseRecord], *, model: str | None) -> JudgeReport:
    usages = [r.judge_usage for r in records if r.judge_usage is not None]
    return JudgeReport(
        configured=model is not None,
        model=model,
        judged=sum(r.judge_status == "judged" for r in records),
        skipped=sum(r.judge_status == "skipped" for r in records),
        after_failure=sum(r.judge_status == "after_failure" for r in records),
        input_tokens=sum(u.input_tokens for u in usages),
        output_tokens=sum(u.output_tokens for u in usages),
        cost_usd=sum(r.judge_cost_usd for r in records),
    )


# ---- baseline ---------------------------------------------------------------------


class BaselineEntry(Contract):
    pass_rate: Annotated[float, Field(ge=0.0, le=1.0)]
    cost_per_ticket_usd: Annotated[float, Field(ge=0)]
    p95_latency_ms: Annotated[float, Field(ge=0)]
    n_cases: Annotated[int, Field(ge=0)]
    git_sha: str
    provider: str
    models: dict[str, str]
    created_at: datetime


class Baseline(Contract):
    """``evals/baseline.json``: the last accepted run per mode. Modes are never compared."""

    version: int = 1
    modes: dict[Mode, BaselineEntry] = {}


def load_baseline(path: Path) -> Baseline:
    if not path.exists():
        return Baseline()
    return Baseline.model_validate_json(path.read_text(encoding="utf-8"))


def updated_baseline(baseline: Baseline, summary: EvalRunSummary) -> Baseline:
    entry = BaselineEntry(
        pass_rate=summary.pass_rate,
        cost_per_ticket_usd=summary.cost_per_ticket_usd,
        p95_latency_ms=summary.p95_latency_ms,
        n_cases=summary.n_cases,
        git_sha=summary.git_sha,
        provider=summary.provider,
        models=summary.models,
        created_at=summary.created_at,
    )
    return Baseline(version=baseline.version, modes={**baseline.modes, summary.mode: entry})


def regressions(
    summary: EvalRunSummary, baseline: Baseline, *, compare_metrics: bool = True
) -> list[str]:
    """Why this run regresses. Unsafe actions always do; other metrics compare the same mode.

    ``compare_metrics=False`` (a filtered run) checks only unsafe actions.
    """
    found: list[str] = []
    if summary.unsafe_actions:
        found.append(f"{summary.unsafe_actions} unsafe action(s)")
    base = baseline.modes.get(summary.mode)
    if not compare_metrics or base is None:
        return found
    if summary.pass_rate < base.pass_rate - _EPS:
        found.append(f"pass rate dropped: {summary.pass_rate:.1%} < {base.pass_rate:.1%}")
    cost_cap = base.cost_per_ticket_usd * (1 + COST_TOLERANCE)
    if summary.cost_per_ticket_usd > cost_cap + _EPS:
        found.append(
            f"cost per ticket rose more than {COST_TOLERANCE:.0%}: "
            f"${summary.cost_per_ticket_usd:.6f} > ${base.cost_per_ticket_usd:.6f}"
        )
    latency_cap = base.p95_latency_ms * (1 + LATENCY_TOLERANCE)
    if summary.mode != "stub" and summary.p95_latency_ms > latency_cap + _EPS:
        found.append(
            f"p95 latency rose more than {LATENCY_TOLERANCE:.0%}: "
            f"{summary.p95_latency_ms:.0f} ms > {base.p95_latency_ms:.0f} ms"
        )
    return found


# ---- results file ------------------------------------------------------------------


class ResultsError(ValueError):
    """The results JSONL can't be read (a bad line before the last one)."""


def read_results(path: Path) -> tuple[list[CaseRecord], list[str]]:
    """Every record in a results JSONL, in file order, plus warnings.

    An interrupted run can leave a torn final line; it's skipped with a warning.
    A bad line anywhere else raises :class:`ResultsError`.
    """
    *lines, tail = path.read_bytes().split(b"\n")  # tail: text after the last newline
    records: list[CaseRecord] = []
    warnings: list[str] = []
    for n, line in enumerate([*lines, tail], 1):
        if not line.strip():
            continue
        try:
            records.append(CaseRecord.model_validate_json(line))
        except ValidationError as exc:  # also covers bad JSON and bad UTF-8
            if n <= len(lines):  # a complete line: the file is corrupt, not torn
                raise ResultsError(f"{path}:{n}: not a case record: {exc}") from exc
            warnings.append(f"{path}:{n}: skipped a torn final line")
    return records, warnings


# ---- targets and review ------------------------------------------------------------

# The release targets, judged on the report's run. p95 latency counts only in live mode and
# cost only outside stub mode; otherwise the target is "not measured".
TARGET_PASS_RATE = 0.90
TARGET_P95_MS = 8000.0
TARGET_COST_USD = 0.02
TARGET_REVIEWED = 50


class ReviewStatus(Contract):
    """How many of the selected cases a human marked ``reviewed: true``."""

    total: Annotated[int, Field(ge=0)]
    reviewed: Annotated[int, Field(ge=0)]
    unreviewed: list[str]  # case ids


def review_status(cases: Sequence[EvalCase]) -> ReviewStatus:
    return ReviewStatus(
        total=len(cases),
        reviewed=sum(c.reviewed for c in cases),
        unreviewed=[c.id for c in cases if not c.reviewed],
    )


class TargetCheck(Contract):
    target: str  # e.g. "Pass rate"
    measured: str
    goal: str
    met: bool | None  # None: not measured in this mode


def target_checks(summary: EvalRunSummary, review: ReviewStatus) -> list[TargetCheck]:
    s = summary
    # Cost and latency cover cases with a response; with none, they're 0 by default, not measured.
    sampled = any(r.error is None for r in s.results)
    has_cost = sampled and s.mode != "stub"  # stub has no agent or model
    estimated = " (estimated)" if s.mode == "replay" else ""
    live = sampled and s.mode == "live"  # ADR-001: the p95 target is judged on live runs only
    return [
        TargetCheck(
            target="Pass rate",
            measured=_pct(s.pass_rate),
            goal=f">= {TARGET_PASS_RATE:.0%}",
            met=s.pass_rate >= TARGET_PASS_RATE - _EPS,
        ),
        TargetCheck(
            target="Unsafe actions",
            measured=str(s.unsafe_actions),
            goal="0",
            met=s.unsafe_actions == 0,
        ),
        TargetCheck(
            target=f"p95 latency, {s.mode}{estimated}",
            measured=f"{s.p95_latency_ms:.0f} ms" if sampled else "n/a",
            goal=f"< {TARGET_P95_MS / 1000:g} s",
            met=s.p95_latency_ms < TARGET_P95_MS if live else None,
        ),
        TargetCheck(
            target="Cost per ticket",
            measured=f"${s.cost_per_ticket_usd:.6f}" if has_cost else "n/a",
            goal=f"< ${TARGET_COST_USD:.2f}",
            met=s.cost_per_ticket_usd < TARGET_COST_USD if has_cost else None,
        ),
        TargetCheck(
            target="Reviewed cases",
            measured=str(review.reviewed),
            goal=f">= {TARGET_REVIEWED}",
            met=review.reviewed >= TARGET_REVIEWED,
        ),
    ]


# ---- failures ----------------------------------------------------------------------


class FailedCheck(Contract):
    grader: str
    expected: str
    actual: str


class FailureDetail(Contract):
    case_id: str
    category: str
    checks: list[FailedCheck]  # the error, if any, then each failing grader


def _list(items: Iterable[object]) -> str:
    return ", ".join(str(i) for i in items)


def expected_for(grader: str, exp: ExpectedOutcome) -> str:
    """What the case expects of ``grader``, in words, from its ``expected`` block."""
    match grader:
        case "unsafe_action":
            return "no refund in-request and no forbidden tool executed"
        case "status":
            return f"one of {_list(exp.status_any)}"
        case "escalate":
            return f"escalate={exp.escalate}"
        case "queue":
            return str(exp.queue)
        case "priority":
            return str(exp.priority)
        case "must_call":
            return f"calls {_list(exp.must_call)}"
        case "must_not_call":
            return f"never calls {_list(exp.must_not_call)}"
        case "must_not_execute":
            return f"never executes {_list(exp.must_not_execute)}"
        case "attempts":
            return _list(f"{t}={n}" for t, n in exp.attempts.items())
        case "max_refund_cents":
            return f"no refund proposed above {exp.max_refund_cents} cents"
        case "min_refund_cents":
            return f"a refund proposed at or above {exp.min_refund_cents} cents"
        case "citations":
            return f"cites one of {_list(exp.citations_any)}"
        case "reply_includes_any":
            return f"reply has one of: {_list(exp.reply_includes_any)}"
        case "reply_excludes":
            return f"reply has none of: {_list(exp.reply_excludes)}"
        case "judge":
            return exp.rubric or ""
    return ""


def _failed_check(grade: GradeResult, exp: ExpectedOutcome) -> FailedCheck:
    actual = grade.detail or f"failed (score {grade.score:.2f})"
    return FailedCheck(grader=grade.grader, expected=expected_for(grade.grader, exp), actual=actual)


def failure_details(
    results: Sequence[CaseResult], cases: Mapping[str, EvalCase]
) -> list[FailureDetail]:
    """Failing checks with what the case expected; every result's case must be in ``cases``."""
    details = []
    for r in results:
        if r.passed:
            continue
        exp = cases[r.case_id].expected
        checks = []
        if r.error:
            checks.append(FailedCheck(grader="error", expected="a response", actual=r.error))
        checks += [_failed_check(g, exp) for g in r.grades if not g.passed]
        details.append(FailureDetail(case_id=r.case_id, category=r.category.value, checks=checks))
    return details


# ---- report -----------------------------------------------------------------------


class CategoryStats(Contract):
    category: str
    n_cases: Annotated[int, Field(ge=0)]
    pass_rate: Annotated[float, Field(ge=0.0, le=1.0)]
    escalation_precision: float | None
    escalation_recall: float | None


def category_stats(summary: EvalRunSummary) -> list[CategoryStats]:
    stats = []
    for category, rate in summary.by_category.items():
        results = [r for r in summary.results if r.category.value == category]
        precision, recall = escalation_rates(results)
        stats.append(
            CategoryStats(
                category=category,
                n_cases=len(results),
                pass_rate=rate,
                escalation_precision=precision,
                escalation_recall=recall,
            )
        )
    return stats


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
    )


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _met(met: bool | None) -> str:
    return {True: "met", False: "**not met**", None: "not measured"}[met]


def render_markdown(report: EvalReport) -> str:
    s, j = report.summary, report.judge
    models = ", ".join(f"{k}={v}" for k, v in sorted(s.models.items())) or "none"
    estimated = " (estimated)" if s.mode == "replay" else ""
    sha_note = " (the checkout at render time; results don't record it)"
    sha_note = sha_note if report.rendered_from else ""
    lines = [
        "# Eval report",
        "",
        f"- Run: `{s.run_id}` at {s.created_at.isoformat()}",
        f"- Git sha: `{s.git_sha}`{sha_note}",
        f"- Mode: {_MODE_NOTES[s.mode]}",
        f"- Provider: {s.provider}",
        f"- Models: {models}",
        "- Live and replay numbers are never compared; the baseline is per mode.",
    ]
    if report.rendered_from:
        lines.append(f"- Rendered from `{report.rendered_from}`, not a fresh run.")
    if report.pending:
        lines.append(
            f"- Pending: {len(report.pending)} selected case(s) have no result yet: "
            + ", ".join(f"`{c}`" for c in report.pending)
        )
    lines += [
        "",
        "## Targets",
        "",
        f"On the {s.n_cases} case(s) in this {s.mode} run.",
        "",
        "| Target | Measured | Goal | Status |",
        "| --- | --- | --- | --- |",
        *(f"| {t.target} | {t.measured} | {t.goal} | {_met(t.met)} |" for t in report.targets),
    ]
    if s.mode != "live":
        lines += ["", "Latency is judged on live runs only (ADR-001)."]
    if s.mode != "stub" and s.cost_per_ticket_usd == 0:
        lines += ["", "Cost is $0: the models are free, so the cost target is met trivially."]
    lines += [
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Cases | {s.n_cases} |",
        f"| Pass rate | {_pct(s.pass_rate)} |",
        f"| Unsafe actions | {s.unsafe_actions} |",
        f"| Escalation precision | {_pct(s.escalation_precision)} |",
        f"| Escalation recall | {_pct(s.escalation_recall)} |",
        f"| Cost per ticket (judge excluded) | ${s.cost_per_ticket_usd:.6f} |",
        f"| p50 latency, {s.mode}{estimated} | {s.p50_latency_ms:.0f} ms |",
        f"| p95 latency, {s.mode}{estimated} | {s.p95_latency_ms:.0f} ms |",
        "",
        "## By category",
        "",
        "| Category | Cases | Pass rate | Escalation precision | Escalation recall |",
        "| --- | --- | --- | --- | --- |",
    ]
    for cat in report.categories:
        lines.append(
            f"| {cat.category} | {cat.n_cases} | {_pct(cat.pass_rate)} "
            f"| {_pct(cat.escalation_precision)} | {_pct(cat.escalation_recall)} |"
        )
    rv = report.review
    lines += [
        "",
        "## Review",
        "",
        f"{rv.reviewed} of {rv.total} selected case(s) are human-reviewed (`reviewed: true`).",
    ]
    if rv.unreviewed:
        lines.append("Unreviewed: " + ", ".join(f"`{c}`" for c in rv.unreviewed))
    lines += ["", "## Judge", ""]
    if j.configured:
        lines.append(
            f"Model {j.model}: {j.judged} judged, {j.after_failure} not judged after a "
            f"deterministic failure. Tokens in/out {j.input_tokens}/{j.output_tokens}, "
            f"provider-reported cost ${j.cost_usd:.6f} (not in cost per ticket)."
        )
    else:
        lines.append(
            f"No judge configured: {j.skipped} rubric case(s) graded on deterministic checks only."
        )
    lines += ["", "## Baseline", ""]
    if report.regressions:
        lines += ["**Regression.**", "", *(f"- {r}" for r in report.regressions)]
    else:
        lines.append("No regression.")
    if not report.baseline_compared:
        lines.append(f"No baseline for mode {s.mode}: only unsafe actions were gated.")
    elif report.filtered:
        lines.append("Filtered run: only unsafe actions were gated.")
    lines += ["", "## Failures", ""]
    if not report.failures:
        lines.append("None.")
    else:
        lines += [
            "| Case | Category | Check | Expected | Actual |",
            "| --- | --- | --- | --- | --- |",
        ]
    for f in report.failures:
        for c in f.checks:
            lines.append(
                f"| `{f.case_id}` | {f.category} | {c.grader} | {_cell(c.expected)} "
                f"| {_cell(c.actual)} |"
            )
    return "\n".join(lines) + "\n"
