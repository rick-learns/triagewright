"""Aggregation: case records into an :class:`EvalRunSummary`, per-category stats, judge usage.

Cost per ticket and latency cover cases that produced a response (no ``error``); errored
cases count against the pass rate instead. Judge calls are never part of cost per ticket.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Annotated

from pydantic import Field

from triagewright.contracts.common import Contract, stable_id
from triagewright.contracts.evals import CaseResult, EvalRunSummary
from triagewright.evals.runner import CaseRecord, Mode


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
