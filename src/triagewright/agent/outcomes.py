"""How code reads an executor outcome: a policy block, a refusal, or a question for the
customer."""

from __future__ import annotations

from triagewright.contracts.agents import ActionOutcome, ActionStatus
from triagewright.contracts.policy import REFUND_OUTCOME
from triagewright.contracts.tools import ErrorCategory, RefundReason

# Errors the customer can fix by giving other details; anything else is a tool failure.
_ASK_CUSTOMER = frozenset(
    {ErrorCategory.NOT_FOUND, ErrorCategory.VALIDATION, ErrorCategory.PERMISSION}
)


def _refund_reason(outcome: ActionOutcome) -> RefundReason | None:
    """The winning reason of a policy-blocked refund (reasons are in precedence order)."""
    elig = outcome.eligibility
    if outcome.status is ActionStatus.BLOCKED and elig is not None and elig.reasons:
        return elig.reasons[0]
    return None


def _refused(outcome: ActionOutcome) -> bool:
    winning = _refund_reason(outcome)
    return winning is not None and REFUND_OUTCOME[winning] is None


def _needs_info(outcome: ActionOutcome) -> bool:
    """A proposal that failed on its arguments, scope, or a missing record, not on the backend."""
    error = outcome.result.error if outcome.result else None
    if error is None or error.retryable or error.category not in _ASK_CUSTOMER:
        return False
    return outcome.status is ActionStatus.FAILED or (
        outcome.status is ActionStatus.BLOCKED and outcome.verdict is None
    )
