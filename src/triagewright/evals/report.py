"""Aggregation into :class:`EvalRunSummary`, the baseline gate, and the markdown/JSON report.

Cost per ticket and latency cover cases that produced a response (no
``error``); errored cases count against the pass rate instead. Judge calls are
never part of cost per ticket; the report lists them separately.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Annotated

from pydantic import Field

from triagewright.contracts.common import Contract, stable_id
from triagewright.contracts.evals import EvalRunSummary
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
    true_pos = sum(r.escalated and r.expected_escalate for r in results)
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
        escalation_precision=_ratio(true_pos, sum(r.escalated for r in results)),
        escalation_recall=_ratio(true_pos, sum(r.expected_escalate for r in results)),
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


# ---- report -----------------------------------------------------------------------


class EvalReport(Contract):
    """The JSON report: the contract summary plus judge usage and the baseline verdict."""

    summary: EvalRunSummary
    judge: JudgeReport
    baseline_compared: bool  # False when the baseline has no entry for this mode
    filtered: bool  # a case filter was used, so only unsafe actions were gated
    regressions: list[str]


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def render_markdown(report: EvalReport) -> str:
    s, j = report.summary, report.judge
    models = ", ".join(f"{k}={v}" for k, v in sorted(s.models.items())) or "none"
    estimated = " (estimated)" if s.mode == "replay" else ""
    lines = [
        "# Eval report",
        "",
        f"- Run: `{s.run_id}` at {s.created_at.isoformat()} on `{s.git_sha}`",
        f"- Mode: {_MODE_NOTES[s.mode]}",
        f"- Provider: {s.provider}",
        f"- Models: {models}",
        "- Live and replay numbers are never compared; the baseline is per mode.",
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
        "| Category | Pass rate |",
        "| --- | --- |",
        *(f"| {c} | {_pct(rate)} |" for c, rate in s.by_category.items()),
        "",
        "## Judge",
        "",
    ]
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
    failed = [r for r in s.results if not r.passed]
    lines += ["", "## Failures", ""]
    if not failed:
        lines.append("None.")
    for r in failed:
        why = [r.error] if r.error else []
        why += [f"{g.grader}: {g.detail}" for g in r.grades if not g.passed]
        lines.append(f"- `{r.case_id}` ({r.category.value}): {'; '.join(why)}")
    return "\n".join(lines) + "\n"
