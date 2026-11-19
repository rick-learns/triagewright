"""Scripted outcomes over the fake case runtime. Each script plays one agent run."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Literal

from triagewright.contracts.agents import (
    ActionOutcome,
    EscalationDecision,
    FinalResponse,
    ProposedAction,
    ResolutionStatus,
)
from triagewright.contracts.common import JsonObject, stable_id
from triagewright.contracts.evals import CATEGORY_PREFIX, EvalCase, ToolFault
from triagewright.contracts.guardrails import ApprovalDecision
from triagewright.contracts.rag import Citation
from triagewright.contracts.telemetry import RequestMetrics
from triagewright.contracts.tickets import Ticket
from triagewright.contracts.tools import CallContext, Caller, ToolName, ToolResult
from triagewright.contracts.world import EscalationQueue, EscalationReason, Priority, WorldState
from triagewright.evals.judge import JudgeClientFactory
from triagewright.evals.runner import CaseRecord, JudgeConfig, Mode, run_case
from triagewright.testing import (
    FakeCaseRuntime,
    FakeCaseRuntimeFactory,
    sample_ticket,
    seed_as_of,
    seed_world,
)

Script = Callable[[FakeCaseRuntime, Ticket], Awaitable[FinalResponse]]

_PREFIX_TO_CATEGORY = {prefix: cat for cat, prefix in CATEGORY_PREFIX.items()}


class ScriptedResolver:
    def __init__(self, runtime: FakeCaseRuntime, scripts: Mapping[str, Script]) -> None:
        self.runtime = runtime
        self.scripts = scripts

    async def resolve(self, ticket: Ticket) -> FinalResponse:
        return await self.scripts[ticket.ticket_id](self.runtime, ticket)


def scripted_factory(scripts: Mapping[str, Script]) -> FakeCaseRuntimeFactory:
    return FakeCaseRuntimeFactory(lambda rt: ScriptedResolver(rt, scripts))


def make_case(
    case_id: str = "happy-001",
    *,
    ticket: Ticket | None = None,
    faults: Sequence[ToolFault] = (),
    world: WorldState | None = None,
    **expected: Any,
) -> EvalCase:
    return EvalCase.model_validate(
        {
            "id": case_id,
            "category": _PREFIX_TO_CATEGORY[case_id.split("-")[0]],
            "title": f"case {case_id}",
            "ticket": ticket or sample_ticket(),
            "world": world,
            "faults": list(faults),
            "expected": {"status_any": ["resolved"], "escalate": False, **expected},
        }
    )


def ctx(ticket: Ticket, caller: Caller = "action") -> CallContext:
    return CallContext(
        request_id=stable_id("req", ticket.ticket_id),
        ticket_id=ticket.ticket_id,
        customer_id=ticket.customer_id,
        caller=caller,
    )


def respond(
    ticket: Ticket,
    status: ResolutionStatus = ResolutionStatus.RESOLVED,
    *,
    reply: str = "Your order is on its way.",
    citations: Sequence[str] = (),
    escalation: EscalationDecision | None = None,
    actions: Sequence[ActionOutcome] = (),
    mode: Literal["live", "replay", "stub"] = "replay",
    metrics: RequestMetrics | None = None,
) -> FinalResponse:
    if metrics is None and mode != "stub":
        metrics = RequestMetrics(
            request_id=stable_id("req", ticket.ticket_id),
            ticket_id=ticket.ticket_id,
            estimated=mode == "replay",
        )
    stored = escalation is not None and escalation.escalate
    return FinalResponse(
        request_id=stable_id("req", ticket.ticket_id),
        ticket_id=ticket.ticket_id,
        status=status,
        reply=reply,
        citations=[
            Citation(doc_id=d, chunk_id=f"{d}#0", title=d, quote="quoted policy") for d in citations
        ],
        escalation=escalation,
        handoff_id=stable_id("esc", ticket.ticket_id) if stored else None,  # the fake gateway's
        actions=list(actions),
        mode=mode,
        metrics=metrics,
    )


async def call(
    rt: FakeCaseRuntime, ticket: Ticket, tool: ToolName, args: JsonObject, caller: Caller = "action"
) -> ToolResult:
    return await rt.gateway.call(tool, args, ctx(ticket, caller))


async def escalate(
    rt: FakeCaseRuntime,
    ticket: Ticket,
    queue: EscalationQueue = EscalationQueue.TIER1,
    priority: Priority = Priority.NORMAL,
) -> EscalationDecision:
    await call(
        rt,
        ticket,
        ToolName.CREATE_ESCALATION,
        {
            "ticket_id": ticket.ticket_id,
            "queue": queue.value,
            "priority": priority.value,
            "summary": "Customer asked for a human.",
            "reason_codes": ["customer_requested_human"],
        },
        caller="coordinator",
    )
    return EscalationDecision(
        escalate=True,
        reasons=[EscalationReason.CUSTOMER_REQUESTED_HUMAN],
        queue=queue,
        priority=priority,
        handoff_summary="Customer asked for a human.",
    )


async def propose_refund(
    rt: FakeCaseRuntime, ticket: Ticket, order_id: str, amount_cents: int | str
) -> ActionOutcome:
    action = ProposedAction(
        tool=ToolName.ISSUE_REFUND,
        arguments={"order_id": order_id, "amount_cents": amount_cents, "reason": "damaged"},
        rationale="customer asked",
    )
    return await rt.executor.propose(action, ctx(ticket), injection_suspected=False)


async def refund_in_request(
    rt: FakeCaseRuntime, ticket: Ticket, order_id: str, amount_cents: int
) -> ActionOutcome:
    """The unsafe path: propose, then approve and execute inside the same request."""
    outcome = await propose_refund(rt, ticket, order_id, amount_cents)
    assert outcome.approval_id is not None
    decision = ApprovalDecision(
        approval_id=outcome.approval_id, approved=True, reviewer="bot", decided_at=seed_as_of()
    )
    return await rt.executor.decide(decision)


async def run(
    case: EvalCase,
    script: Script,
    *,
    mode: Mode = "replay",
    timeout_s: float = 5.0,
    judge: JudgeConfig | None = None,
) -> CaseRecord:
    factory = scripted_factory({case.ticket.ticket_id: script})
    return await run_case(
        factory,
        case,
        seed_world=seed_world(),
        seed_as_of=seed_as_of(),
        mode=mode,
        timeout_s=timeout_s,
        judge=judge,
    )


def judge_config(factory: JudgeClientFactory, recordings_dir: Any) -> JudgeConfig:
    return JudgeConfig(factory=factory, model="judge-model", recordings_dir=recordings_dir)


def grades_by_name(record: CaseRecord) -> dict[str, bool]:
    return {g.grader: g.passed for g in record.result.grades}
