"""A fake stack for the API and CLI tests, built only from ``triagewright.testing``."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from triagewright.api.services import Services
from triagewright.contracts.agents import ActionStatus, ProposedAction
from triagewright.contracts.tools import CallContext, ToolName
from triagewright.testing import (
    FakeActionExecutor,
    FakeToolGateway,
    InMemoryApprovalStore,
    PassthroughGuardrails,
    StubResolver,
    seed_as_of,
    seed_world,
)

TOKEN = "approver-test-token"


@dataclass
class Stack:
    services: Services
    gateway: FakeToolGateway
    approvals: InMemoryApprovalStore
    approval_id: str  # a pending, eligible refund for cus_hal08


def fake_stack() -> Stack:
    gateway = FakeToolGateway(seed_world(), as_of=seed_as_of())
    approvals = InMemoryApprovalStore()
    executor = FakeActionExecutor(gateway, PassthroughGuardrails(), approvals, now=seed_as_of())
    proposal = ProposedAction(
        tool=ToolName.ISSUE_REFUND,
        arguments={"order_id": "ord_h8001", "amount_cents": 5900, "reason": "damaged"},
        rationale="damaged on arrival",
    )
    ctx = CallContext(
        request_id="req_00000001", ticket_id="tkt_0001", customer_id="cus_hal08", caller="action"
    )
    outcome = asyncio.run(executor.propose(proposal, ctx, injection_suspected=False))
    assert outcome.status is ActionStatus.PENDING_APPROVAL and outcome.approval_id
    services = Services(resolver=StubResolver(), executor=executor, approvals=approvals)
    return Stack(services, gateway, approvals, outcome.approval_id)
