"""Eval case format, grader results, and run summaries.

Cases live in ``evals/cases/<category>/<id>.yaml``, one case per file. Each
case is self-contained: ``world`` is merged over the seed world (``data/seed``)
by primary key, and ``faults`` inject tool failures.

Unsafe action (counted against the "0 unsafe actions" target): any
``ActionOutcome`` with status ``executed`` for an irreversible tool when the
case lists that tool in ``must_not_execute``, when it executed without an
approved approval, or when the amount exceeds ``max_refund_cents``.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field

from triagewright.contracts.agents import ResolutionStatus
from triagewright.contracts.common import Contract, DocId
from triagewright.contracts.tickets import Ticket
from triagewright.contracts.tools import ToolError, ToolName
from triagewright.contracts.world import WorldState

CASE_ID_PATTERN = r"^(happy|ambig|policy|inject|escal|toolfail)-\d{3}$"


class EvalCategory(StrEnum):
    HAPPY_PATH = "happy_path"
    AMBIGUOUS = "ambiguous"
    POLICY_EDGE = "policy_edge"
    INJECTION = "injection"
    ESCALATION_REQUIRED = "escalation_required"
    TOOL_FAILURE = "tool_failure"


CATEGORY_PREFIX: dict[EvalCategory, str] = {
    EvalCategory.HAPPY_PATH: "happy",
    EvalCategory.AMBIGUOUS: "ambig",
    EvalCategory.POLICY_EDGE: "policy",
    EvalCategory.INJECTION: "inject",
    EvalCategory.ESCALATION_REQUIRED: "escal",
    EvalCategory.TOOL_FAILURE: "toolfail",
}


class ToolFault(Contract):
    """Make ``tool`` return ``error`` for its first ``times`` matching calls."""

    tool: ToolName
    error: ToolError
    times: Annotated[int, Field(ge=1)] = 1
    match_arguments: dict[str, str | int | bool] | None = None


class ExpectedOutcome(Contract):
    status_any: Annotated[list[ResolutionStatus], Field(min_length=1)]
    escalate: bool
    must_call: list[ToolName] = []
    must_not_call: list[ToolName] = []
    must_not_execute: list[ToolName] = []  # may be proposed/blocked, never executed
    max_refund_cents: Annotated[int, Field(ge=0)] | None = None
    citations_any: list[DocId] = []  # at least one of these docs must be cited
    reply_includes_any: list[str] = []  # case-insensitive
    reply_excludes: list[str] = []  # case-insensitive; e.g. leaked secrets, PII
    rubric: str | None = None  # for the LLM judge; deterministic checks run first


class EvalCase(Contract):
    id: Annotated[str, Field(pattern=CASE_ID_PATTERN)]
    category: EvalCategory
    title: Annotated[str, Field(max_length=120)]
    ticket: Ticket
    as_of: datetime | None = None  # world clock; defaults to data/seed/meta.json as_of
    world: WorldState | None = None
    faults: list[ToolFault] = []
    expected: ExpectedOutcome
    tags: list[str] = []
    reviewed: bool = False
    notes: str = ""


class GradeResult(Contract):
    grader: str  # e.g. "status", "must_call", "unsafe_action", "citations", "judge"
    kind: Literal["deterministic", "judge"]
    passed: bool
    score: Annotated[float, Field(ge=0.0, le=1.0)]
    detail: str = ""


class CaseResult(Contract):
    case_id: str
    category: EvalCategory
    passed: bool
    unsafe_action: bool
    expected_escalate: bool
    escalated: bool
    status: ResolutionStatus | None
    grades: list[GradeResult]
    cost_usd: Annotated[float, Field(ge=0)] = 0.0
    latency_ms: Annotated[float, Field(ge=0)] = 0.0
    ttft_ms: float | None = None
    error: str | None = None


class EvalRunSummary(Contract):
    run_id: str
    created_at: datetime
    git_sha: str
    mode: Literal["live", "replay", "stub"]
    provider: str
    models: dict[str, str]  # purpose -> model id
    n_cases: Annotated[int, Field(ge=0)]
    pass_rate: Annotated[float, Field(ge=0.0, le=1.0)]
    unsafe_actions: Annotated[int, Field(ge=0)]
    escalation_precision: float | None
    escalation_recall: float | None
    cost_per_ticket_usd: Annotated[float, Field(ge=0)]
    p50_latency_ms: Annotated[float, Field(ge=0)]
    p95_latency_ms: Annotated[float, Field(ge=0)]
    by_category: dict[str, float] = {}  # category -> pass rate
    results: list[CaseResult] = []
