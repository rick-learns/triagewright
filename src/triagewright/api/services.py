"""What the API and CLI share: the ports they run on, the wiring hook, and the decision path.

Both surfaces are thin. They take a :class:`Services` bundle of ports and never
build lanes themselves. The real bundle comes from integration
(``triagewright.wiring.build_services(settings)``, task X1); until that lands,
:func:`load_services` raises :class:`WiringUnavailableError` with a helpful message.
"""

from __future__ import annotations

import hmac
import importlib
from dataclasses import dataclass
from datetime import datetime

from pydantic import SecretStr, ValidationError

from triagewright.config import Settings
from triagewright.contracts.agents import ActionOutcome
from triagewright.contracts.guardrails import ApprovalDecision, ApprovalStatus
from triagewright.contracts.ports import ActionExecutor, ApprovalStore, TicketResolver

WIRING_MODULE = "triagewright.wiring"


@dataclass(frozen=True)
class Services:
    resolver: TicketResolver
    executor: ActionExecutor
    approvals: ApprovalStore


class WiringUnavailableError(RuntimeError):
    """The real lanes aren't wired yet (``triagewright.wiring`` lands at integration)."""


class ApprovalNotFoundError(LookupError):
    pass


class DecisionConflictError(Exception):
    pass


def load_services(settings: Settings) -> Services:
    """Placeholder wiring hook: ``triagewright.wiring.build_services(settings)`` if it exists."""
    try:
        wiring = importlib.import_module(WIRING_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name != WIRING_MODULE:
            raise
        wiring = None
    build = getattr(wiring, "build_services", None)
    if build is None:
        raise WiringUnavailableError(
            "the agent isn't wired yet: triagewright.wiring.build_services(settings) arrives "
            "with integration (task X1). Until then, use create_app(...) or create_cli(...) "
            "with your own Services, e.g. the fakes in triagewright.testing."
        )
    services = build(settings)
    if not isinstance(services, Services):
        raise WiringUnavailableError("triagewright.wiring.build_services must return Services")
    return services


def token_matches(presented: str | None, expected: SecretStr) -> bool:
    """Constant-time comparison of a presented approver token with ``TW_APPROVER_TOKEN``."""
    if not presented:
        return False
    return hmac.compare_digest(presented.encode(), expected.get_secret_value().encode())


async def decide_approval(
    services: Services,
    approval_id: str,
    *,
    approved: bool,
    reviewer: str,
    note: str,
    now: datetime,
) -> ActionOutcome:
    """Record an authenticated human decision through ``ActionExecutor.decide``.

    Raises :class:`ApprovalNotFoundError` for an unknown id and :class:`DecisionConflictError`
    for a repeat decision the executor would refuse. Repeating an approve for an
    approved, not yet executed request resumes execution, as the port specifies.
    """
    current = services.approvals.get(approval_id)
    if current is None:
        raise ApprovalNotFoundError(approval_id)
    resumable = approved and current.status is ApprovalStatus.APPROVED
    if current.status is not ApprovalStatus.PENDING and not resumable:
        raise DecisionConflictError(f"{approval_id} is already {current.status.value}")
    decision = ApprovalDecision(
        approval_id=current.approval_id,
        approved=approved,
        reviewer=reviewer,
        note=note,
        decided_at=now,
    )
    try:
        return await services.executor.decide(decision)
    except ValidationError:
        raise
    except ValueError as exc:  # another decision landed between the check and the write
        raise DecisionConflictError(f"{approval_id} was decided concurrently") from exc
