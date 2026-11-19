"""The tool handlers over :class:`MockSystems`. Shared by the MCP server and the in-process gateway.

Behavior matches ``triagewright.testing.FakeToolGateway`` (the reference), in
the same order: scope (``UNSCOPED_TOOLS``), argument validation, then the
tool. Another customer's records look like ``not_found``. ``create_escalation``
needs ``caller="coordinator"`` and ``ctx.ticket_id``, and is idempotent per
ticket. ``issue_refund`` checks, in order, the HMAC approval token and
``caller="human"``, ownership, the idempotency key (a committed refund
returns success), then ``policy.evaluate_refund``, in one transaction.

:meth:`ToolService.call` never raises: failures become ``ToolResult`` errors.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from typing import Any

from pydantic import ValidationError

from triagewright.contracts.common import JsonObject, stable_id
from triagewright.contracts.guardrails import PolicyConfig
from triagewright.contracts.policy import evaluate_refund, verify_approval_token
from triagewright.contracts.ports import Retriever
from triagewright.contracts.rag import MetadataFilter, RetrievalQuery
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
    ToolName,
    ToolResult,
)
from triagewright.contracts.world import Escalation, Refund
from triagewright.mcp_server.store import MockSystems

logger = logging.getLogger(__name__)


class ToolService:
    """Runs one tool call against the mock systems, with the world clock pinned to ``as_of``."""

    def __init__(
        self,
        systems: MockSystems,
        *,
        as_of: datetime,
        approval_key: bytes,
        retriever: Retriever | None = None,
        policy: PolicyConfig | None = None,
    ) -> None:
        self.systems = systems
        self.as_of = as_of
        self.retriever = retriever
        self.policy = policy or PolicyConfig()
        self._approval_key = approval_key

    async def call(self, name: ToolName, arguments: JsonObject, ctx: CallContext) -> ToolResult:
        try:
            return await self._call(name, arguments, ctx)
        except sqlite3.OperationalError as exc:  # e.g. "database is locked"
            logger.warning("tool %s: database error: %s", name, exc)
            return ToolResult.fail(name, ErrorCategory.TRANSIENT, "database busy")
        except Exception:
            logger.exception("tool %s failed", name)
            return ToolResult.fail(name, ErrorCategory.INTERNAL, "internal error")

    async def _call(self, name: ToolName, arguments: JsonObject, ctx: CallContext) -> ToolResult:
        if ctx.customer_id is None and name not in UNSCOPED_TOOLS:
            return ToolResult.fail(name, ErrorCategory.PERMISSION, "no customer in scope")
        try:
            args = TOOL_SPECS[name].input_model.model_validate(arguments)
        except ValidationError as exc:
            return ToolResult.fail(name, ErrorCategory.VALIDATION, str(exc.errors()[0]["msg"]))
        return await self._dispatch(name, args, ctx)

    async def _dispatch(self, name: ToolName, args: Any, ctx: CallContext) -> ToolResult:
        missing = ErrorCategory.NOT_FOUND
        mine = ctx.customer_id
        db = self.systems
        match args:
            case GetCustomerInput(customer_id=cid):
                c = db.customer(cid) if cid == mine else None
                return ToolResult.ok(name, c) if c else ToolResult.fail(name, missing, cid)
            case GetAccountInput(customer_id=cid):
                a = db.account(cid) if cid == mine else None
                return ToolResult.ok(name, a) if a else ToolResult.fail(name, missing, cid)
            case GetOrderInput(order_id=oid):
                o = db.order(oid)
                if o and o.customer_id == mine:
                    return ToolResult.ok(name, o)
                return ToolResult.fail(name, missing, oid)
            case ListOrdersInput(customer_id=cid, limit=limit):
                if cid != mine or db.customer(cid) is None:
                    return ToolResult.fail(name, missing, cid)
                orders = sorted(db.orders_of(cid), key=lambda o: o.placed_at, reverse=True)
                return ToolResult.ok(name, OrderList(orders=orders[:limit]))
            case SearchKBInput() as req:
                return ToolResult.ok(name, await self._search(req))
            case CheckRefundEligibilityInput(order_id=oid, amount_cents=amount):
                elig = self._eligibility(oid, amount, ctx)
                return ToolResult.ok(name, elig) if elig else ToolResult.fail(name, missing, oid)
            case IssueRefundInput() as req:
                return self._issue_refund(name, req, ctx)
            case CreateEscalationInput() as req:
                return self._create_escalation(name, req, ctx)
        return ToolResult.fail(name, ErrorCategory.INTERNAL, "unhandled tool")  # pragma: no cover

    async def _search(self, req: SearchKBInput) -> KBSearchResult:
        if self.retriever is None:
            return KBSearchResult(hits=[])
        flt = MetadataFilter(categories=req.categories) if req.categories else None
        found = await self.retriever.retrieve(RetrievalQuery(text=req.query, k=req.k, filters=flt))
        return KBSearchResult(
            hits=[
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
        )

    def _eligibility(
        self, order_id: str, amount: int, ctx: CallContext
    ) -> RefundEligibility | None:
        db = self.systems
        order = db.order(order_id)
        customer = db.customer(ctx.customer_id) if ctx.customer_id else None
        if order is None or customer is None or order.customer_id != customer.customer_id:
            return None
        return evaluate_refund(
            order=order,
            customer=customer,
            refunds=db.refunds_of(customer.customer_id),
            amount_cents=amount,
            as_of=self.as_of,
            policy=self.policy,
        )

    def _create_escalation(
        self, name: ToolName, req: CreateEscalationInput, ctx: CallContext
    ) -> ToolResult:
        if ctx.caller != "coordinator" or req.ticket_id != ctx.ticket_id:
            return ToolResult.fail(name, ErrorCategory.PERMISSION, "ticket out of scope")
        with self.systems.transaction():
            existing = self.systems.escalation_for(req.ticket_id)
            if existing is not None:
                return ToolResult.ok(name, existing)  # idempotent per ticket
            esc = Escalation(
                escalation_id=stable_id("esc", req.ticket_id),
                ticket_id=req.ticket_id,
                queue=req.queue,
                priority=req.priority,
                summary=req.summary,
                reason_codes=req.reason_codes,
                created_at=self.as_of,
            )
            self.systems.add_escalation(esc)
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
        db = self.systems
        with db.transaction():
            # Order: ownership, then idempotency (a committed retry succeeds), then policy.
            order = db.order(req.order_id)
            if order is None or order.customer_id != ctx.customer_id:
                return ToolResult.fail(name, ErrorCategory.NOT_FOUND, req.order_id)
            prior = db.refund_by_key(req.idempotency_key)
            if prior is not None:
                if (prior.order_id, prior.amount_cents) == (req.order_id, req.amount_cents):
                    return ToolResult.ok(name, prior)
                return ToolResult.fail(name, ErrorCategory.CONFLICT, "idempotency key reused")
            elig = self._eligibility(req.order_id, req.amount_cents, ctx)
            if elig is None:  # pragma: no cover - ownership was checked above
                return ToolResult.fail(name, ErrorCategory.NOT_FOUND, req.order_id)
            if not elig.eligible:
                return ToolResult.fail(name, ErrorCategory.POLICY, ", ".join(elig.reasons))
            refund = Refund(
                refund_id=stable_id("rf", req.idempotency_key),
                order_id=req.order_id,
                customer_id=order.customer_id,
                amount_cents=req.amount_cents,
                reason=req.reason,
                idempotency_key=req.idempotency_key,
                approval_id=req.approval_id,
                created_at=self.as_of,
            )
            db.add_refund(refund)
        return ToolResult.ok(name, refund)
