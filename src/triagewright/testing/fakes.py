"""In-memory fakes for every port. Orchestrator-owned; lanes test against these.

They are deliberately simple and deterministic. They are not the real
implementations: the MCP server (lane A), retriever (lane C), guardrails
(lane D), and telemetry (lane E) replace them at integration.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import ValidationError

from triagewright.contracts.agents import (
    ActionOutcome,
    ActionStatus,
    EscalationDecision,
    FinalResponse,
    ProposedAction,
    ResolutionStatus,
)
from triagewright.contracts.common import JsonObject, canonical_json, stable_id
from triagewright.contracts.evals import ToolFault
from triagewright.contracts.guardrails import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalStatus,
    ContentSource,
    GuardrailDecision,
    GuardrailVerdict,
    PolicyConfig,
    PolicyContext,
    RedactionResult,
)
from triagewright.contracts.llm import ModelRequest, ModelResponse
from triagewright.contracts.policy import (
    approval_token,
    evaluate_refund,
    refund_idempotency_key,
    verify_approval_token,
)
from triagewright.contracts.ports import Guardrails, Retriever, TicketResolver
from triagewright.contracts.rag import Chunk, KBDocument, RetrievalQuery, RetrievedChunk
from triagewright.contracts.telemetry import AttrValue, RequestMetrics, SpanKind
from triagewright.contracts.tickets import Ticket
from triagewright.contracts.tools import (
    CODE_OWNED_TOOLS,
    IRREVERSIBLE_TOOLS,
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
    RefundProposal,
    SearchKBInput,
    ToolCallRecord,
    ToolName,
    ToolResult,
    ToolSpec,
    proposal_model,
)
from triagewright.contracts.world import (
    Escalation,
    EscalationQueue,
    EscalationReason,
    Priority,
    Refund,
    WorldState,
)

# Shared by the fake gateway and fake executor. The real key comes from TW_APPROVAL_KEY.
TEST_APPROVAL_KEY = b"triagewright-test-approval-key"

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
                    trusted=doc.trusted,
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


class PassthroughGuardrails:
    """No redaction, no screening, and a minimal action policy (lane D replaces it).

    Blocks writes once injection is suspected and refunds without a passing
    eligibility check. Every other irreversible action needs approval.
    """

    def redact(self, text: str) -> RedactionResult:
        return RedactionResult(text=text)

    def screen(self, text: str, source: ContentSource) -> GuardrailVerdict:
        return GuardrailVerdict(decision=GuardrailDecision.ALLOW)

    def check_action(self, action: ProposedAction, ctx: PolicyContext) -> GuardrailVerdict:
        if ctx.injection_suspected and action.tool in IRREVERSIBLE_TOOLS:
            return GuardrailVerdict(
                decision=GuardrailDecision.BLOCK,
                rule_ids=["injection.write_blocked"],
                injection_suspected=True,
            )
        if action.tool is ToolName.ISSUE_REFUND:
            elig = ctx.eligibility
            if elig is None or not elig.eligible:
                reasons = [r.value for r in elig.reasons] if elig else ["no eligibility check"]
                return GuardrailVerdict(
                    decision=GuardrailDecision.BLOCK,
                    rule_ids=[f"refund.{r}" for r in reasons],
                    reasons=reasons,
                )
        if action.tool in IRREVERSIBLE_TOOLS:
            return GuardrailVerdict(
                decision=GuardrailDecision.REQUIRE_APPROVAL, rule_ids=["approval.irreversible"]
            )
        return GuardrailVerdict(decision=GuardrailDecision.ALLOW)


class InMemoryApprovalStore:
    def __init__(self) -> None:
        self.items: dict[str, ApprovalRequest] = {}

    def submit(self, request: ApprovalRequest) -> ApprovalRequest:
        return self.items.setdefault(request.approval_id, request)

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


class FakeActionExecutor:
    """Reference implementation of the ``ActionExecutor`` port over the fakes."""

    def __init__(
        self,
        gateway: FakeToolGateway,
        guardrails: Guardrails,
        approvals: InMemoryApprovalStore,
        *,
        now: datetime,
        approval_key: bytes = TEST_APPROVAL_KEY,
        tool_retries: int = 2,
    ) -> None:
        self.gateway = gateway
        self.guardrails = guardrails
        self.approvals = approvals
        self.now = now
        self._key = approval_key
        self._retries = tool_retries

    async def _call_with_retries(
        self, name: ToolName, arguments: JsonObject, ctx: CallContext
    ) -> tuple[ToolResult, int]:
        attempts = 0
        while True:
            attempts += 1
            result = await self.gateway.call(name, arguments, ctx)
            retryable = result.error is not None and result.error.retryable
            if not retryable or attempts > self._retries:
                return result, attempts

    async def propose(
        self, action: ProposedAction, ctx: CallContext, *, injection_suspected: bool
    ) -> ActionOutcome:
        if action.tool in CODE_OWNED_TOOLS:
            verdict = GuardrailVerdict(
                decision=GuardrailDecision.BLOCK, rule_ids=["tool.code_owned"]
            )
            return ActionOutcome(action=action, status=ActionStatus.BLOCKED, verdict=verdict)
        try:
            args = proposal_model(action.tool).model_validate(action.arguments)
        except ValidationError as exc:
            result = ToolResult.fail(
                action.tool, ErrorCategory.VALIDATION, str(exc.errors()[0]["msg"])
            )
            return ActionOutcome(action=action, status=ActionStatus.BLOCKED, result=result)
        arguments: JsonObject = args.model_dump(mode="json")
        approval_id = stable_id("apr", ctx.ticket_id, action.tool, canonical_json(arguments))
        if action.tool in IRREVERSIBLE_TOOLS:
            if injection_suspected:
                verdict = GuardrailVerdict(
                    decision=GuardrailDecision.BLOCK,
                    rule_ids=["injection.write_blocked"],
                    injection_suspected=True,
                )
                return ActionOutcome(action=action, status=ActionStatus.BLOCKED, verdict=verdict)
            existing = self.approvals.get(approval_id)
            if existing is not None and existing.customer_id == ctx.customer_id:
                return ActionOutcome(
                    action=action,
                    status=_STORED_STATUS[existing.status],
                    verdict=existing.verdict,
                    approval_id=approval_id,
                )
        elig = None
        if isinstance(args, RefundProposal):
            check, attempts = await self._call_with_retries(
                ToolName.CHECK_REFUND_ELIGIBILITY,
                {"order_id": args.order_id, "amount_cents": args.amount_cents},
                ctx,
            )
            if check.is_error or check.data is None:
                return ActionOutcome(
                    action=action, status=ActionStatus.FAILED, result=check, attempts=attempts
                )
            elig = RefundEligibility.model_validate(check.data)
        policy_ctx = PolicyContext(
            ticket_id=ctx.ticket_id,
            customer_id=ctx.customer_id,
            eligibility=elig,
            injection_suspected=injection_suspected,
            now=self.now,
        )
        verdict = self.guardrails.check_action(action, policy_ctx)
        blocked = ActionOutcome(
            action=action, status=ActionStatus.BLOCKED, verdict=verdict, eligibility=elig
        )
        if verdict.decision is GuardrailDecision.BLOCK:
            return blocked
        if verdict.decision is GuardrailDecision.REQUIRE_APPROVAL:
            if action.tool not in IRREVERSIBLE_TOOLS or ctx.customer_id is None:
                return blocked
            stored = self.approvals.submit(
                ApprovalRequest(
                    approval_id=approval_id,
                    ticket_id=ctx.ticket_id,
                    customer_id=ctx.customer_id,
                    tool=action.tool,
                    arguments=arguments,
                    idempotency_key=refund_idempotency_key(approval_id),
                    rationale=action.rationale,
                    verdict=verdict,
                    requested_at=self.now,
                )
            )
            return ActionOutcome(
                action=action,
                status=_STORED_STATUS[stored.status],
                verdict=verdict,
                approval_id=approval_id,
                eligibility=elig,
            )
        if action.tool in IRREVERSIBLE_TOOLS:  # a guardrail bug must not run a refund
            return blocked
        result = await self.gateway.call(action.tool, arguments, ctx)
        status = ActionStatus.FAILED if result.is_error else ActionStatus.EXECUTED
        return ActionOutcome(
            action=action, status=status, result=result, verdict=verdict, attempts=1
        )

    async def decide(self, decision: ApprovalDecision) -> ActionOutcome:
        current = self.approvals.get(decision.approval_id)
        if current is not None and current.status is ApprovalStatus.APPROVED and decision.approved:
            return await self.execute_approved(current.approval_id)  # resume after a failure
        request = self.approvals.decide(decision)  # raises on any other repeat decision
        if request.status is ApprovalStatus.APPROVED:
            return await self.execute_approved(request.approval_id)
        return ActionOutcome(
            action=_stored_action(request),
            status=ActionStatus.SKIPPED,
            approval_id=request.approval_id,
        )

    async def execute_approved(self, approval_id: str) -> ActionOutcome:
        request = self.approvals.get(approval_id)
        if request is None:
            raise KeyError(approval_id)
        action = _stored_action(request)
        if request.status is not ApprovalStatus.APPROVED:
            return ActionOutcome(
                action=action, status=ActionStatus.BLOCKED, approval_id=approval_id
            )
        proposal = RefundProposal.model_validate(request.arguments)
        token = approval_token(
            self._key,
            approval_id=approval_id,
            order_id=proposal.order_id,
            amount_cents=proposal.amount_cents,
            idempotency_key=request.idempotency_key,
        )
        ctx = CallContext(
            request_id=stable_id("req", request.ticket_id, "approval", approval_id),
            ticket_id=request.ticket_id,
            customer_id=request.customer_id,
            caller="human",
        )
        arguments: JsonObject = {
            **request.arguments,
            "idempotency_key": request.idempotency_key,
            "approval_id": approval_id,
            "approval_token": token,
        }
        result = await self.gateway.call(request.tool, arguments, ctx)
        if result.is_error:
            return ActionOutcome(
                action=action,
                status=ActionStatus.FAILED,
                result=result,
                approval_id=approval_id,
                attempts=1,
            )
        self.approvals.mark_executed(approval_id)
        return ActionOutcome(
            action=action,
            status=ActionStatus.EXECUTED,
            result=result,
            approval_id=approval_id,
            attempts=1,
        )


# What re-proposing an existing approval reports. Never "executed" in-request.
_STORED_STATUS: dict[ApprovalStatus, ActionStatus] = {
    ApprovalStatus.PENDING: ActionStatus.PENDING_APPROVAL,
    ApprovalStatus.APPROVED: ActionStatus.PENDING_APPROVAL,
    ApprovalStatus.REJECTED: ActionStatus.BLOCKED,
    ApprovalStatus.EXECUTED: ActionStatus.SKIPPED,
}


def _stored_action(request: ApprovalRequest) -> ProposedAction:
    return ProposedAction(
        tool=request.tool, arguments=request.arguments, rationale=request.rationale[:500]
    )


@dataclass
class FakeCaseRuntime:
    """One isolated fake stack. Satisfies the ``CaseRuntime`` port."""

    gateway: FakeToolGateway
    approvals: InMemoryApprovalStore
    executor: FakeActionExecutor
    resolver: TicketResolver = field(default_factory=lambda: StubResolver())

    def tool_calls(self) -> list[ToolCallRecord]:
        return list(self.gateway.calls)

    def ledger(self) -> WorldState:
        return self.gateway.ledger()

    def pending_approvals(self) -> list[ApprovalRequest]:
        return self.approvals.list_pending()


class FakeCaseRuntimeFactory:
    """Builds a fresh fake stack per case. ``make_resolver`` wires a resolver onto it."""

    def __init__(
        self,
        make_resolver: Callable[[FakeCaseRuntime], TicketResolver] | None = None,
        *,
        docs: Sequence[KBDocument] = (),
        policy: PolicyConfig | None = None,
    ) -> None:
        self.built: list[str] = []  # case ids, in build order
        self._make_resolver = make_resolver
        self._retriever = FakeRetriever(docs) if docs else None
        self._policy = policy

    def build(
        self, *, case_id: str, world: WorldState, as_of: datetime, faults: Sequence[ToolFault]
    ) -> FakeCaseRuntime:
        self.built.append(case_id)
        gateway = FakeToolGateway(
            world, as_of=as_of, retriever=self._retriever, faults=faults, policy=self._policy
        )
        approvals = InMemoryApprovalStore()
        executor = FakeActionExecutor(gateway, PassthroughGuardrails(), approvals, now=as_of)
        runtime = FakeCaseRuntime(gateway=gateway, approvals=approvals, executor=executor)
        if self._make_resolver is not None:
            runtime.resolver = self._make_resolver(runtime)
        return runtime


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
        self.paced_ms = 0.0

    @contextmanager
    def span(
        self, name: str, kind: SpanKind, attributes: Mapping[str, AttrValue] | None = None
    ) -> Iterator[_NullSpan]:
        self.spans.append((name, kind))
        yield _NullSpan()

    def record_model_call(
        self,
        request: ModelRequest,
        response: ModelResponse,
        started_ms: float,
        *,
        paced_ms: float = 0.0,
    ) -> None:
        self.responses.append(response)

    def record_pacing(self, paced_ms: float) -> None:
        self.paced_ms += paced_ms

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
            request_id=stable_id("req", ticket.ticket_id),
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
