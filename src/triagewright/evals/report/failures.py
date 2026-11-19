"""Failure details: each failing check paired with what the case expected, in words."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from triagewright.contracts.common import Contract
from triagewright.contracts.evals import CaseResult, EvalCase, ExpectedOutcome, GradeResult


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
