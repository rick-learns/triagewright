"""Call contexts and fake gateway/executor builders shared by the fake tests."""

from __future__ import annotations

from collections.abc import Sequence

from triagewright.contracts.evals import ToolFault
from triagewright.contracts.tools import CallContext, Caller
from triagewright.testing import (
    FakeActionExecutor,
    FakeRetriever,
    FakeToolGateway,
    InMemoryApprovalStore,
    PassthroughGuardrails,
    seed_as_of,
    seed_kb,
    seed_world,
)


def ctx(customer_id: str | None = "cus_ava01", caller: Caller = "action") -> CallContext:
    return CallContext(
        request_id="req_00000001", ticket_id="tkt_0001", customer_id=customer_id, caller=caller
    )


CTX = ctx()


def gateway(faults: Sequence[ToolFault] = ()) -> FakeToolGateway:
    return FakeToolGateway(
        seed_world(),
        as_of=seed_as_of(),
        retriever=FakeRetriever(seed_kb()),
        faults=faults,
    )


def executor(gw: FakeToolGateway | None = None) -> FakeActionExecutor:
    return FakeActionExecutor(
        gw or gateway(), PassthroughGuardrails(), InMemoryApprovalStore(), now=seed_as_of()
    )
