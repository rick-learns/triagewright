"""In-memory fakes for every port. Orchestrator-owned; lanes test against these.

They are deliberately simple and deterministic. They are not the real
implementations: the MCP server (lane A), retriever (lane C), guardrails
(lane D), and telemetry (lane E) replace them at integration.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any

from pydantic import ValidationError

from triagewright.contracts.agents import (
    EscalationDecision,
    EscalationReason,
    FinalResponse,
    ProposedAction,
    ResolutionStatus,
)
from triagewright.contracts.common import JsonObject
from triagewright.contracts.evals import ToolFault
from triagewright.contracts.guardrails import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalStatus,
    ContentSource,
    GuardrailDecision,
    GuardrailVerdict,
    PolicyContext,
    RedactionResult,
)
from triagewright.contracts.llm import ModelRequest, ModelResponse
from triagewright.contracts.rag import Chunk, KBDocument, RetrievalQuery, RetrievedChunk
from triagewright.contracts.telemetry import AttrValue, RequestMetrics, SpanKind
from triagewright.contracts.tickets import Ticket
from triagewright.contracts.tools import (
    IRREVERSIBLE_TOOLS,
    TOOL_SPECS,
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
    ToolSpec,
)
from triagewright.contracts.world import (
    Escalation,
    EscalationQueue,
    Order,
    OrderStatus,
    Priority,
    Refund,
    WorldState,
)

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return {t for t in _WORD.findall(text.lower()) if len(t) > 2}


def chunk_documents(docs: Sequence[KBDocument]) -> list[Chunk]:
    """Split on blank lines. A stand-in for lane C's chunker."""
    chunks: list[Chunk] = []
    for doc in docs:
        pos = 0
        ordinal = 0
        for para in doc.body.split("\n\n"):
            start = doc.body.index(para, pos)
            pos = start + len(para)
            if not para.strip() or para.lstrip().startswith("#"):
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#{ordinal}",
                    doc_id=doc.doc_id,
                    ordinal=ordinal,
                    title=doc.title,
                    category=doc.category,
                    tags=doc.tags,
                    text=para.strip(),
                    start_char=start,
                    end_char=pos,
                )
            )
            ordinal += 1
    return chunks


class FakeRetriever:
    """Keyword-overlap retriever over paragraph chunks."""

    def __init__(self, docs: Sequence[KBDocument]) -> None:
        self.chunks = chunk_documents(docs)

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]:
        q = _tokens(query.text)
        scored: list[tuple[float, Chunk]] = []
        for chunk in self.chunks:
            f = query.filters
            if f and f.categories and chunk.category not in f.categories:
                continue
            if f and f.tags_any and not set(f.tags_any) & set(chunk.tags):
                continue
            words = _tokens(chunk.text + " " + chunk.title)
            score = len(q & words) / (len(q) or 1)
            if score > max(query.min_score, 0.0):
                scored.append((score, chunk))
        scored.sort(key=lambda s: (-s[0], s[1].chunk_id))
        return [
            RetrievedChunk(chunk=c, score=s, rank=i + 1, method="keyword")
            for i, (s, c) in enumerate(scored[: query.k])
        ]


class FakeToolGateway:
    """Mock business systems in memory, with fault injection. Mirrors the MCP tool contract."""

    def __init__(
        self,
        world: WorldState,
        *,
        as_of: datetime,
        retriever: FakeRetriever | None = None,
        faults: Sequence[ToolFault] = (),
        refund_window_days: int = 30,
    ) -> None:
        self.customers = {c.customer_id: c for c in world.customers}
        self.accounts = {a.customer_id: a for a in world.accounts}
        self.orders = {o.order_id: o for o in world.orders}
        self.refunds: list[Refund] = list(world.refunds)
        self.escalations: list[Escalation] = list(world.escalations)
        self.as_of = as_of
        self.retriever = retriever
        self.refund_window = timedelta(days=refund_window_days)
        self._faults = [[f, f.times] for f in faults]
        self.calls: list[tuple[ToolName, JsonObject]] = []

    def list_tools(self) -> Sequence[ToolSpec]:
        return list(TOOL_SPECS.values())

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
        self.calls.append((name, dict(arguments)))
        if (faulted := self._fault_for(name, arguments)) is not None:
            return faulted
        spec = TOOL_SPECS[name]
        try:
            args = spec.input_model.model_validate(arguments)
        except ValidationError as exc:
            return ToolResult.fail(name, ErrorCategory.VALIDATION, str(exc.errors()[0]["msg"]))
        return await self._dispatch(name, args, ctx)

    def _refunded(self, order_id: str) -> int:
        return sum(r.amount_cents for r in self.refunds if r.order_id == order_id)

    def _eligibility(self, order: Order, amount: int) -> RefundEligibility:
        reasons: list[str] = []
        window_end = order.delivered_at + self.refund_window if order.delivered_at else None
        if order.status is not OrderStatus.DELIVERED:
            reasons.append(f"order status is {order.status.value}")
        if window_end and self.as_of > window_end:
            reasons.append("outside refund window")
        already = self._refunded(order.order_id)
        max_refundable = max(order.total_cents - already, 0)
        if amount > max_refundable:
            reasons.append("amount exceeds refundable balance")
        return RefundEligibility(
            order_id=order.order_id,
            eligible=not reasons,
            max_refundable_cents=max_refundable,
            already_refunded_cents=already,
            window_ends_at=window_end,
            reasons=reasons,
        )

    async def _dispatch(self, name: ToolName, args: Any, ctx: CallContext) -> ToolResult:
        missing = ErrorCategory.NOT_FOUND
        match args:
            case GetCustomerInput(customer_id=cid):
                c = self.customers.get(cid)
                return ToolResult.ok(name, c) if c else ToolResult.fail(name, missing, cid)
            case GetAccountInput(customer_id=cid):
                a = self.accounts.get(cid)
                return ToolResult.ok(name, a) if a else ToolResult.fail(name, missing, cid)
            case GetOrderInput(order_id=oid):
                o = self.orders.get(oid)
                return ToolResult.ok(name, o) if o else ToolResult.fail(name, missing, oid)
            case ListOrdersInput(customer_id=cid, limit=limit):
                if cid not in self.customers:
                    return ToolResult.fail(name, missing, cid)
                mine = sorted(
                    (o for o in self.orders.values() if o.customer_id == cid),
                    key=lambda o: o.placed_at,
                    reverse=True,
                )
                return ToolResult.ok(name, OrderList(orders=mine[:limit]))
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
                            text=r.chunk.text,
                            score=r.score,
                        )
                        for r in found
                    ]
                return ToolResult.ok(name, KBSearchResult(hits=hits))
            case CheckRefundEligibilityInput(order_id=oid, amount_cents=amount):
                o = self.orders.get(oid)
                if not o:
                    return ToolResult.fail(name, missing, oid)
                return ToolResult.ok(name, self._eligibility(o, amount))
            case IssueRefundInput() as req:
                return self._issue_refund(name, req)
            case CreateEscalationInput() as req:
                esc = Escalation(
                    escalation_id=f"esc_{len(self.escalations) + 1:04d}",
                    ticket_id=req.ticket_id,
                    queue=req.queue,
                    priority=req.priority,
                    summary=req.summary,
                    reason_codes=req.reason_codes,
                    created_at=self.as_of,
                )
                self.escalations.append(esc)
                return ToolResult.ok(name, esc)
        return ToolResult.fail(name, ErrorCategory.INTERNAL, "unhandled tool")  # pragma: no cover

    def _issue_refund(self, name: ToolName, req: IssueRefundInput) -> ToolResult:
        for r in self.refunds:
            if r.idempotency_key == req.idempotency_key:
                same = (r.order_id, r.amount_cents) == (req.order_id, req.amount_cents)
                if same:
                    return ToolResult.ok(name, r)
                return ToolResult.fail(name, ErrorCategory.CONFLICT, "idempotency key reused")
        if req.approval_id is None:
            return ToolResult.fail(name, ErrorCategory.PERMISSION, "approval_id required")
        order = self.orders.get(req.order_id)
        if not order:
            return ToolResult.fail(name, ErrorCategory.NOT_FOUND, req.order_id)
        elig = self._eligibility(order, req.amount_cents)
        if not elig.eligible:
            return ToolResult.fail(name, ErrorCategory.POLICY, "; ".join(elig.reasons))
        refund = Refund(
            refund_id=f"rf_{len(self.refunds) + 1:04d}",
            order_id=order.order_id,
            customer_id=order.customer_id,
            amount_cents=req.amount_cents,
            reason=req.reason,
            idempotency_key=req.idempotency_key,
            approval_id=req.approval_id,
            created_at=self.as_of,
        )
        self.refunds.append(refund)
        return ToolResult.ok(name, refund)


class PassthroughGuardrails:
    """No redaction, no screening. Irreversible tools always need approval (safe default)."""

    def redact(self, text: str) -> RedactionResult:
        return RedactionResult(text=text)

    def screen(self, text: str, source: ContentSource) -> GuardrailVerdict:
        return GuardrailVerdict(decision=GuardrailDecision.ALLOW)

    def check_action(self, action: ProposedAction, ctx: PolicyContext) -> GuardrailVerdict:
        if action.tool in IRREVERSIBLE_TOOLS:
            return GuardrailVerdict(
                decision=GuardrailDecision.REQUIRE_APPROVAL, rule_ids=["approval.irreversible"]
            )
        return GuardrailVerdict(decision=GuardrailDecision.ALLOW)


class InMemoryApprovalStore:
    def __init__(self) -> None:
        self.items: dict[str, ApprovalRequest] = {}

    def submit(self, request: ApprovalRequest) -> ApprovalRequest:
        self.items[request.approval_id] = request
        return request

    def get(self, approval_id: str) -> ApprovalRequest | None:
        return self.items.get(approval_id)

    def decide(self, decision: ApprovalDecision) -> ApprovalRequest:
        current = self.items[decision.approval_id]
        if current.status is not ApprovalStatus.PENDING:
            raise ValueError(f"{decision.approval_id} is already {current.status.value}")
        status = ApprovalStatus.APPROVED if decision.approved else ApprovalStatus.REJECTED
        updated = current.model_copy(update={"status": status})
        self.items[decision.approval_id] = updated
        return updated

    def mark_executed(self, approval_id: str) -> ApprovalRequest:
        current = self.items[approval_id]
        if current.status is not ApprovalStatus.APPROVED:
            raise ValueError(f"{approval_id} is {current.status.value}, not approved")
        updated = current.model_copy(update={"status": ApprovalStatus.EXECUTED})
        self.items[approval_id] = updated
        return updated

    def list_pending(self) -> list[ApprovalRequest]:
        return [a for a in self.items.values() if a.status is ApprovalStatus.PENDING]


class ScriptedModelClient:
    """Returns pre-built responses in order and records every request."""

    def __init__(self, responses: Sequence[ModelResponse]) -> None:
        self._responses = list(responses)
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if not self._responses:
            raise RuntimeError("ScriptedModelClient ran out of responses")
        return self._responses.pop(0)


class _NullSpan:
    def __init__(self) -> None:
        self.attributes: dict[str, AttrValue] = {}
        self.error: str | None = None

    def set_attribute(self, key: str, value: AttrValue) -> None:
        self.attributes[key] = value

    def set_error(self, message: str) -> None:
        self.error = message


class NullTelemetry:
    """Counts calls and tokens; costs are always 0. Records span names for assertions."""

    def __init__(self, request_id: str = "req_00000000", ticket_id: str = "tkt_0000") -> None:
        self.request_id = request_id
        self.ticket_id = ticket_id
        self.spans: list[tuple[str, SpanKind]] = []
        self.responses: list[ModelResponse] = []
        self.tool_calls = 0

    @contextmanager
    def span(
        self, name: str, kind: SpanKind, attributes: Mapping[str, AttrValue] | None = None
    ) -> Iterator[_NullSpan]:
        self.spans.append((name, kind))
        yield _NullSpan()

    def record_model_call(self, request: ModelRequest, response: ModelResponse) -> None:
        self.responses.append(response)

    def record_tool_call(self, result: ToolResult, latency_ms: float) -> None:
        self.tool_calls += 1

    def finish(self) -> RequestMetrics:
        return RequestMetrics(
            request_id=self.request_id,
            ticket_id=self.ticket_id,
            tool_calls=self.tool_calls,
            input_tokens=sum(r.usage.input_tokens for r in self.responses),
            output_tokens=sum(r.usage.output_tokens for r in self.responses),
        )


class NullTelemetryFactory:
    def for_request(self, request_id: str, ticket_id: str) -> NullTelemetry:
        return NullTelemetry(request_id, ticket_id)


class StubResolver:
    """Escalates every ticket. Lets API/CLI/evals run before the agent lands."""

    async def resolve(self, ticket: Ticket) -> FinalResponse:
        return FinalResponse(
            request_id="req_stub0000",
            ticket_id=ticket.ticket_id,
            status=ResolutionStatus.ESCALATED,
            reply="Thanks for reaching out. A support specialist will follow up shortly.",
            escalation=EscalationDecision(
                escalate=True,
                reasons=[EscalationReason.OUT_OF_SCOPE],
                queue=EscalationQueue.TIER1,
                priority=Priority.NORMAL,
                handoff_summary="Stub resolver: no agent wired yet.",
            ),
            mode="stub",
        )
