"""Scripted responses and a fake stack for coordinator tests."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from triagewright.agent.coordinator import PURPOSES, Coordinator
from triagewright.contracts.agents import LoopConfig
from triagewright.contracts.common import JsonObject
from triagewright.contracts.evals import ToolFault
from triagewright.contracts.guardrails import (
    ContentSource,
    GuardrailDecision,
    GuardrailVerdict,
    RedactionResult,
)
from triagewright.contracts.llm import ModelRequest, ModelResponse, StopReason, ToolResultBlock
from triagewright.contracts.ports import Guardrails
from triagewright.contracts.tickets import Ticket
from triagewright.contracts.tools import ToolCallRecord, ToolName
from triagewright.contracts.world import WorldState
from triagewright.testing import (
    FakeActionExecutor,
    FakeRetriever,
    FakeToolGateway,
    InMemoryApprovalStore,
    NullTelemetry,
    PassthroughGuardrails,
    ScriptedModelClient,
    model_response,
    seed_as_of,
    seed_kb,
    seed_world,
    text_block,
    tool_use_block,
)

# ---- scripted model responses ----------------------------------------------------


def call(name: str, arguments: JsonObject | None = None) -> ModelResponse:
    return model_response(tool_use_block(name, arguments or {}))


def submit(subagent: str, payload: JsonObject) -> ModelResponse:
    return call(f"submit_{subagent}", payload)


def text(message: str, stop: StopReason = StopReason.END_TURN) -> ModelResponse:
    return model_response(text_block(message), stop_reason=stop)


def triage(
    category: str = "order_status",
    *,
    intent: str = "Find out where the order is",
    urgency: str = "normal",
    needs_human: bool = False,
    reasons: Sequence[str] = (),
    injection: bool = False,
) -> ModelResponse:
    return submit(
        "triage",
        {
            "category": category,
            "intent": intent,
            "urgency": urgency,
            "needs_human": needs_human,
            "needs_human_reasons": list(reasons),
            "injection_suspected": injection,
            "confidence": 0.9,
        },
    )


def finalize(
    reply: str = "Your order is on its way.",
    status: str = "resolved",
    citations: Sequence[tuple[str, str]] = (),
) -> ModelResponse:
    cites: list[Any] = [{"chunk_id": c, "quote": q} for c, q in citations]
    return call("finalize", {"status": status, "reply": reply, "citations": cites})


def review(
    escalate: bool = False,
    queue: str | None = None,
    reasons: Sequence[str] = (),
    priority: str = "normal",
    summary: str = "",
) -> ModelResponse:
    return submit(
        "escalation",
        {
            "escalate": escalate,
            "reasons": list(reasons),
            "queue": queue,
            "priority": priority,
            "handoff_summary": summary,
        },
    )


def propose_refund(order_id: str, amount_cents: int) -> ModelResponse:
    return call(
        "propose_action",
        {
            "tool": "issue_refund",
            "arguments": {"order_id": order_id, "amount_cents": amount_cents, "reason": "Asked"},
            "rationale": "The customer asked for a refund.",
        },
    )


REFUND_POLICY = "kb_refund_policy#0"


def refund_script(order_id: str, amount_cents: int, **review_kw: Any) -> list[ModelResponse]:
    """Triage, a policy search, the action subagent proposing one refund, finalize, and review."""
    return [
        triage("refund_request", intent="Get a refund"),
        call("search_kb", {"query": "refund policy"}),
        call("action"),
        propose_refund(order_id, amount_cents),
        submit("action", {"summary": "Proposed a refund."}),
        finalize(
            "Thanks, I've looked into your refund request.",
            citations=[(REFUND_POLICY, "Delivered orders can be refunded within 30 days")],
        ),
        review(**review_kw),
    ]


# ---- fakes -------------------------------------------------------------------------


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class TimedModel:
    """A scripted model that advances a fake clock, or hangs, on chosen calls."""

    def __init__(
        self,
        responses: Sequence[ModelResponse],
        *,
        clock: FakeClock | None = None,
        step_s: float = 0.0,
        hang_on: Sequence[int] = (),
        fail_on: Sequence[int] = (),
    ) -> None:
        self.inner = ScriptedModelClient(responses)
        self.clock = clock
        self.step_s = step_s
        self.hang_on = set(hang_on)
        self.fail_on = set(fail_on)
        self.calls = 0

    @property
    def requests(self) -> list[ModelRequest]:
        return self.inner.requests

    async def complete(self, request: ModelRequest) -> ModelResponse:
        index = self.calls
        self.calls += 1
        if index in self.hang_on:
            await asyncio.sleep(10)
        if index in self.fail_on:
            raise RuntimeError("provider unavailable")
        if self.clock is not None:
            self.clock.now += self.step_s
        return await self.inner.complete(request)


class FlaggingGuardrails(PassthroughGuardrails):
    """Flags injection when a pattern appears in content from one of ``sources``."""

    def __init__(
        self, *patterns: str, sources: Sequence[ContentSource] = tuple(ContentSource)
    ) -> None:
        self.patterns = [p.lower() for p in patterns]
        self.sources = set(sources)
        self.screened: list[tuple[ContentSource, str]] = []

    def screen(self, text: str, source: ContentSource) -> GuardrailVerdict:
        self.screened.append((source, text))
        if source in self.sources and any(p in text.lower() for p in self.patterns):
            return GuardrailVerdict(
                decision=GuardrailDecision.BLOCK,
                rule_ids=["injection.override"],
                injection_suspected=True,
            )
        return GuardrailVerdict(decision=GuardrailDecision.ALLOW)


class RedactingGuardrails(PassthroughGuardrails):
    def __init__(self, secrets: dict[str, str]) -> None:
        self.secrets = secrets

    def redact(self, text: str) -> RedactionResult:
        for secret, placeholder in self.secrets.items():
            text = text.replace(secret, placeholder)
        return RedactionResult(text=text)


class CapturingTelemetryFactory:
    def __init__(self) -> None:
        self.last: NullTelemetry | None = None

    def for_request(self, request_id: str, ticket_id: str) -> NullTelemetry:
        self.last = NullTelemetry(request_id, ticket_id)
        return self.last


@dataclass
class Stack:
    coordinator: Coordinator
    model: ScriptedModelClient | TimedModel
    gateway: FakeToolGateway
    approvals: InMemoryApprovalStore
    telemetry: CapturingTelemetryFactory
    sleeps: list[float] = field(default_factory=list)

    async def resolve(self, ticket: Ticket) -> Any:
        return await self.coordinator.resolve(ticket)

    def calls(self, tool: ToolName | None = None) -> list[ToolCallRecord]:
        return [c for c in self.gateway.calls if tool is None or c.tool is tool]

    def escalation_args(self) -> JsonObject:
        (record,) = self.calls(ToolName.CREATE_ESCALATION)
        assert record.caller == "coordinator"
        return record.arguments

    def requests(self, purpose: str | None = None) -> list[ModelRequest]:
        return [r for r in self.model.requests if purpose is None or r.purpose == purpose]


def stack(
    responses: Sequence[ModelResponse] = (),
    *,
    model: ScriptedModelClient | TimedModel | None = None,
    guardrails: Guardrails | None = None,
    faults: Sequence[ToolFault] = (),
    config: LoopConfig | None = None,
    clock: FakeClock | None = None,
    world: WorldState | None = None,
) -> Stack:
    gateway = FakeToolGateway(
        world or seed_world(),
        as_of=seed_as_of(),
        retriever=FakeRetriever(seed_kb()),
        faults=faults,
    )
    guard = guardrails or PassthroughGuardrails()
    approvals = InMemoryApprovalStore()
    executor = FakeActionExecutor(gateway, guard, approvals, now=seed_as_of())
    telemetry = CapturingTelemetryFactory()
    client = model or ScriptedModelClient(responses)
    sleeps: list[float] = []

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    coordinator = Coordinator(
        model=client,
        gateway=gateway,
        guardrails=guard,
        executor=executor,
        telemetry=telemetry,
        models=dict.fromkeys(PURPOSES, "test-model"),
        config=config or LoopConfig(),
        mode="replay",
        sleep=sleep,
        **({"clock": clock} if clock is not None else {}),
    )
    return Stack(coordinator, client, gateway, approvals, telemetry, sleeps)


def tool_results(request: ModelRequest) -> list[ToolResultBlock]:
    return [
        block
        for message in request.messages
        for block in message.content
        if isinstance(block, ToolResultBlock)
    ]


def request_text(request: ModelRequest) -> str:
    return request.model_dump_json()
