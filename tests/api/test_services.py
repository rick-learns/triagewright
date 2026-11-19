from __future__ import annotations

import sys
import types

import pytest
from pydantic import SecretStr

from tests.api.stack import Stack, fake_stack
from triagewright.api.services import (
    WIRING_MODULE,
    DecisionConflictError,
    Services,
    WiringUnavailableError,
    decide_approval,
    load_services,
    token_matches,
)
from triagewright.config import Settings
from triagewright.contracts.agents import ActionOutcome, ActionStatus
from triagewright.contracts.guardrails import ApprovalDecision, ApprovalStatus
from triagewright.testing import seed_as_of

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def test_token_matches_only_the_exact_token() -> None:
    expected = SecretStr("s3cret")
    assert token_matches("s3cret", expected)
    assert not token_matches("s3cret ", expected)
    assert not token_matches("", expected)
    assert not token_matches(None, expected)


def test_load_services_uses_wiring_when_it_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    services = fake_stack().services
    wiring = types.ModuleType(WIRING_MODULE)
    wiring.build_services = lambda settings: services  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, WIRING_MODULE, wiring)
    assert load_services(Settings(_env_file=None)) is services
    wiring.build_services = lambda settings: object()  # type: ignore[attr-defined]
    with pytest.raises(WiringUnavailableError, match="must return Services"):
        load_services(Settings(_env_file=None))


async def test_approve_resumes_after_a_failed_execution(stack: Stack) -> None:
    order = stack.gateway.orders.pop("ord_h8001")  # the refund call now fails with not_found
    failed = await decide_approval(
        stack.services, stack.approval_id, approved=True, reviewer="t", note="", now=seed_as_of()
    )
    assert failed.status is ActionStatus.FAILED
    assert stack.approvals.items[stack.approval_id].status is ApprovalStatus.APPROVED
    with pytest.raises(DecisionConflictError, match="already approved"):
        await decide_approval(
            stack.services,
            stack.approval_id,
            approved=False,
            reviewer="t",
            note="",
            now=seed_as_of(),
        )
    stack.gateway.orders["ord_h8001"] = order
    resumed = await decide_approval(
        stack.services, stack.approval_id, approved=True, reviewer="t", note="", now=seed_as_of()
    )
    assert resumed.status is ActionStatus.EXECUTED


async def test_a_concurrent_decision_is_a_conflict(stack: Stack) -> None:

    class Racing:
        async def propose(self, *args: object, **kwargs: object) -> ActionOutcome:
            raise NotImplementedError

        async def decide(self, decision: ApprovalDecision) -> ActionOutcome:
            raise ValueError(f"{decision.approval_id} is already rejected")

        async def execute_approved(self, approval_id: str) -> ActionOutcome:
            raise NotImplementedError

    services = Services(stack.services.resolver, Racing(), stack.approvals)
    with pytest.raises(DecisionConflictError, match="decided concurrently"):
        await decide_approval(
            services, stack.approval_id, approved=True, reviewer="t", note="", now=seed_as_of()
        )
