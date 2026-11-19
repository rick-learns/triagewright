"""Builders and inspectors shared by the guarded executor tests."""

from __future__ import annotations

from collections.abc import Sequence

from triagewright.contracts.agents import LoopConfig, ProposedAction
from triagewright.contracts.evals import ToolFault
from triagewright.contracts.guardrails import ApprovalDecision
from triagewright.contracts.tools import CallContext, Caller, ErrorCategory, ToolError, ToolName
from triagewright.guardrails import GuardedActionExecutor, RuleGuardrails, SqliteApprovalStore
from triagewright.testing import TEST_APPROVAL_KEY, FakeToolGateway, seed_as_of, seed_world


def ctx(customer_id: str | None = "cus_hal08", caller: Caller = "action") -> CallContext:
    return CallContext(
        request_id="req_00000001", ticket_id="tkt_0001", customer_id=customer_id, caller=caller
    )


def gateway(faults: Sequence[ToolFault] = ()) -> FakeToolGateway:
    return FakeToolGateway(seed_world(), as_of=seed_as_of(), faults=faults)


def executor(
    gw: FakeToolGateway | None = None, **kwargs: object
) -> tuple[GuardedActionExecutor, FakeToolGateway, SqliteApprovalStore]:
    gw = gw or gateway()
    store = SqliteApprovalStore()
    ex = GuardedActionExecutor(
        gw,
        RuleGuardrails(),
        store,
        approval_key=TEST_APPROVAL_KEY,
        clock=seed_as_of,
        loop=LoopConfig(retry_backoff_s=0),
        **kwargs,  # type: ignore[arg-type]
    )
    return ex, gw, store


def refund(order_id: str = "ord_h8001", amount: int = 5900) -> ProposedAction:
    return ProposedAction(
        tool=ToolName.ISSUE_REFUND,
        arguments={"order_id": order_id, "amount_cents": amount, "reason": "damaged"},
        rationale="damaged on arrival",
    )


def approve(approval_id: str, approved: bool = True) -> ApprovalDecision:
    return ApprovalDecision(
        approval_id=approval_id, approved=approved, reviewer="supervisor", decided_at=seed_as_of()
    )


def fault(tool: ToolName, category: ErrorCategory, times: int = 1, **kw: int) -> ToolFault:
    error = ToolError(category=category, message="x", retryable=category.value != "not_found", **kw)
    return ToolFault(tool=tool, error=error, times=times)


def calls(gw: FakeToolGateway, tool: ToolName) -> list[Caller]:
    return [c.caller for c in gw.calls if c.tool is tool]


def new_refunds(gw: FakeToolGateway) -> int:
    return len(gw.refunds) - len(seed_world().refunds)
