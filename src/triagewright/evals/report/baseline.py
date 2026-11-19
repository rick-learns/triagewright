"""The baseline gate: ``evals/baseline.json`` per mode, and why a run regresses against it."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated

from pydantic import Field

from triagewright.contracts.common import Contract
from triagewright.contracts.evals import EvalRunSummary
from triagewright.evals.runner import Mode

COST_TOLERANCE = 0.10  # cost per ticket may rise at most 10% (same mode)
LATENCY_TOLERANCE = 0.10  # p95 latency may rise at most 10% (same mode)
_EPS = 1e-9


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
