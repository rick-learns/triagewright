"""The release targets and the human-review count, judged on one run."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated

from pydantic import Field

from triagewright.contracts.common import Contract
from triagewright.contracts.evals import EvalCase, EvalRunSummary
from triagewright.evals.report.baseline import _EPS

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


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"
