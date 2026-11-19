"""Shared builders for the ``test_report_*`` modules (no tests here)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from tests.evals.scenarios import make_case
from triagewright.contracts.evals import (
    CaseResult,
    EvalCase,
    EvalCategory,
    EvalRunSummary,
    GradeResult,
)
from triagewright.contracts.llm import Usage
from triagewright.evals.report import summarize
from triagewright.evals.runner import CaseRecord, JudgeStatus, Mode

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


def cases_for(*ids: str, reviewed: int = 0, **expected: Any) -> list[EvalCase]:
    """Cases for ``ids``; the first ``reviewed`` are marked reviewed."""
    return [
        make_case(c, **expected).model_copy(update={"reviewed": n < reviewed})
        for n, c in enumerate(ids)
    ]


def failing(grader: str, detail: str) -> GradeResult:
    return GradeResult(grader=grader, kind="deterministic", passed=False, score=0.0, detail=detail)
