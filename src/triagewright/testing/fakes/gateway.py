"""In-memory tool gateway that mirrors the MCP tool contract, with fault injection."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from pydantic import ValidationError

from triagewright.contracts.common import JsonObject, stable_id
from triagewright.contracts.evals import ToolFault
from triagewright.contracts.guardrails import (
    PolicyConfig,
)
from triagewright.contracts.policy import (
    evaluate_refund,
    verify_approval_token,
)
from triagewright.contracts.ports import Retriever
from triagewright.contracts.rag import RetrievalQuery
from triagewright.contracts.tools import (
    TOOL_SPECS,
    UNSCOPED_TOOLS,
    CallContext,
    CheckRefundEligibilityInput,
    CreateEscalationInput,
    ErrorCategory,
    GetAccountInput,
    GetCustomerInput,
    GetOrderInput,
    IssueRefundInput,
    KBHit,
    KBSearchResult,
    ListOrdersInput,
    OrderList,
    RefundEligibility,
    SearchKBInput,
    ToolCallRecord,
    ToolName,
    ToolResult,
    ToolSpec,
)
from triagewright.contracts.world import (
    Escalation,
    Refund,
    WorldState,
)

# Shared by the fake gateway and fake executor. The real key comes from TW_APPROVAL_KEY.
TEST_APPROVAL_KEY = b"triagewright-test-approval-key"


class FakeToolGateway:
    """Mock business systems in memory, with fault injection. Mirrors the MCP tool contract.

    Like the real server it scopes every private tool to ``ctx.customer_id``
    (another customer's records look missing), applies ``contracts.policy``,
    verifies the approval token on refunds, and logs every raw call.
    """

    def __init__(
        self,
        world: WorldState,
        *,
        as_of: datetime,
        retriever: Retriever | None = None,
        faults: Sequence[ToolFault] = (),
        policy: PolicyConfig | None = None,
        approval_key: bytes = TEST_APPROVAL_KEY,
    ) -> None:
        self.customers = {c.customer_id: c for c in world.customers}
        self.accounts = {a.customer_id: a for a in world.accounts}
        self.orders = {o.order_id: o for o in world.orders}
        self.refunds: list[Refund] = list(world.refunds)
        self.escalations: list[Escalation] = list(world.escalations)
        self.as_of = as_of
        self.retriever = retriever
        self.policy = policy or PolicyConfig()
        self._approval_key = approval_key
        self._faults = [[f, f.times] for f in faults]
        self.calls: list[ToolCallRecord] = []

    def list_tools(self) -> Sequence[ToolSpec]:
        return list(TOOL_SPECS.values())

    def ledger(self) -> WorldState:
        return WorldState(
            customers=list(self.customers.values()),
            accounts=list(self.accounts.values()),
            orders=list(self.orders.values()),
            refunds=list(self.refunds),
            escalations=list(self.escalations),
        )

    def _fault_for(self, name: ToolName, arguments: JsonObject) -> ToolResult | None:
        for entry in self._faults:
            fault, remaining = entry[0], entry[1]
            assert isinstance(fault, ToolFault) and isinstance(remaining, int)
            if fault.tool is not name or remaining <= 0:
                continue
            if fault.match_arguments and any(
                arguments.get(k) != v for k, v in fault.match_arguments.items()
            ):
                continue
            entry[1] = remaining - 1
            return ToolResult(tool=name, is_error=True, error=fault.error)
        return None

    async def call(self, name: ToolName, arguments: JsonObject, ctx: CallContext) -> ToolResult:
        result = await self._call(name, arguments, ctx)
        self.calls.append(
            ToolCallRecord(tool=name, arguments=dict(arguments), caller=ctx.caller, result=result)
        )
        return result

    async def _call(self, name: ToolName, arguments: JsonObject, ctx: CallContext) -> ToolResult:
        if (faulted := self._fault_for(name, arguments)) is not None:
            return faulted
        if ctx.customer_id is None and name not in UNSCOPED_TOOLS:
            return ToolResult.fail(name, ErrorCategory.PERMISSION, "no customer in scope")
        spec = TOOL_SPECS[name]
        try:
            args = spec.input_model.model_validate(arguments)
        except ValidationError as exc:
            return ToolResult.fail(name, ErrorCategory.VALIDATION, str(exc.errors()[0]["msg"]))
        return await self._dispatch(name, args, ctx)

    def _eligibility(
        self, order_id: str, amount: int, ctx: CallContext
    ) -> RefundEligibility | None:
        order = self.orders.get(order_id)
        customer = self.customers.get(ctx.customer_id or "")
        if order is None or customer is None or order.customer_id != customer.customer_id:
            return None
        return evaluate_refund(
            order=order,
            customer=customer,
            refunds=self.refunds,
            amount_cents=amount,
            as_of=self.as_of,
            policy=self.policy,
        )

    async def _dispatch(self, name: ToolName, args: Any, ctx: CallContext) -> ToolResult:
        missing = ErrorCategory.NOT_FOUND
        mine = ctx.customer_id
        match args:
            case GetCustomerInput(customer_id=cid):
                c = self.customers.get(cid) if cid == mine else None
                return ToolResult.ok(name, c) if c else ToolResult.fail(name, missing, cid)
            case GetAccountInput(customer_id=cid):
                a = self.accounts.get(cid) if cid == mine else None
                return ToolResult.ok(name, a) if a else ToolResult.fail(name, missing, cid)
            case GetOrderInput(order_id=oid):
                o = self.orders.get(oid)
                if o and o.customer_id == mine:
                    return ToolResult.ok(name, o)
                return ToolResult.fail(name, missing, oid)
            case ListOrdersInput(customer_id=cid, limit=limit):
                if cid != mine or cid not in self.customers:
                    return ToolResult.fail(name, missing, cid)
                orders = sorted(
                    (o for o in self.orders.values() if o.customer_id == cid),
                    key=lambda o: o.placed_at,
                    reverse=True,
                )
                return ToolResult.ok(name, OrderList(orders=orders[:limit]))
            case SearchKBInput(query=q, k=k, categories=cats):
                from triagewright.contracts.rag import MetadataFilter

                hits: list[KBHit] = []
                if self.retriever:
                    flt = MetadataFilter(categories=cats) if cats else None
                    found = await self.retriever.retrieve(RetrievalQuery(text=q, k=k, filters=flt))
                    hits = [
                        KBHit(
                            doc_id=r.chunk.doc_id,
                            chunk_id=r.chunk.chunk_id,
                            title=r.chunk.title,
                            category=r.chunk.category,
                            trusted=r.chunk.trusted,
                            text=r.chunk.text,
                            score=r.score,
                        )
                        for r in found
                    ]
                return ToolResult.ok(name, KBSearchResult(hits=hits))
            case CheckRefundEligibilityInput(order_id=oid, amount_cents=amount):
                elig = self._eligibility(oid, amount, ctx)
                return ToolResult.ok(name, elig) if elig else ToolResult.fail(name, missing, oid)
            case IssueRefundInput() as req:
                return self._issue_refund(name, req, ctx)
            case CreateEscalationInput() as req:
                return self._create_escalation(name, req, ctx)
        return ToolResult.fail(name, ErrorCategory.INTERNAL, "unhandled tool")  # pragma: no cover

    def _create_escalation(
        self, name: ToolName, req: CreateEscalationInput, ctx: CallContext
    ) -> ToolResult:
        if ctx.caller != "coordinator" or req.ticket_id != ctx.ticket_id:
            return ToolResult.fail(name, ErrorCategory.PERMISSION, "ticket out of scope")
        for esc in self.escalations:
            if esc.ticket_id == req.ticket_id:
                return ToolResult.ok(name, esc)  # idempotent per ticket
        esc = Escalation(
            escalation_id=stable_id("esc", req.ticket_id),
            ticket_id=req.ticket_id,
            queue=req.queue,
            priority=req.priority,
            summary=req.summary,
            reason_codes=req.reason_codes,
            created_at=self.as_of,
        )
        self.escalations.append(esc)
        return ToolResult.ok(name, esc)

    def _issue_refund(self, name: ToolName, req: IssueRefundInput, ctx: CallContext) -> ToolResult:
        signed = verify_approval_token(
            self._approval_key,
            req.approval_token,
            approval_id=req.approval_id,
            order_id=req.order_id,
            amount_cents=req.amount_cents,
            idempotency_key=req.idempotency_key,
        )
        if not signed or ctx.caller != "human":
            return ToolResult.fail(name, ErrorCategory.PERMISSION, "valid approval required")
        # Order: ownership, then idempotency (a committed retry succeeds), then policy.
        order = self.orders.get(req.order_id)
        if order is None or order.customer_id != ctx.customer_id:
            return ToolResult.fail(name, ErrorCategory.NOT_FOUND, req.order_id)
        for r in self.refunds:
            if r.idempotency_key == req.idempotency_key:
                same = (r.order_id, r.amount_cents) == (req.order_id, req.amount_cents)
                if same:
                    return ToolResult.ok(name, r)
                return ToolResult.fail(name, ErrorCategory.CONFLICT, "idempotency key reused")
        elig = self._eligibility(req.order_id, req.amount_cents, ctx)
        if elig is None:  # pragma: no cover - ownership was checked above
            return ToolResult.fail(name, ErrorCategory.NOT_FOUND, req.order_id)
        if not elig.eligible:
            return ToolResult.fail(name, ErrorCategory.POLICY, ", ".join(elig.reasons))
        refund = Refund(
            refund_id=stable_id("rf", req.idempotency_key),
            order_id=req.order_id,
            customer_id=self.orders[req.order_id].customer_id,
            amount_cents=req.amount_cents,
            reason=req.reason,
            idempotency_key=req.idempotency_key,
            approval_id=req.approval_id,
            created_at=self.as_of,
        )
        self.refunds.append(refund)
        return ToolResult.ok(name, refund)
