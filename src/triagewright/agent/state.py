"""Per-request state: what code owns for one ``resolve`` call, its time budget, and its
screens."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from pydantic import BaseModel

from triagewright.agent import prompts
from triagewright.agent.common import Clock, Fallback, Sleep, escape_data
from triagewright.contracts.agents import (
    ActionOutcome,
    AnswerDraft,
    LoopConfig,
    LoopStop,
    ProposedAction,
    TriageResult,
)
from triagewright.contracts.common import canonical_json
from triagewright.contracts.guardrails import ContentSource, GuardrailDecision
from triagewright.contracts.ports import (
    ActionExecutor,
    Guardrails,
    ModelClient,
    RequestTelemetry,
    ToolGateway,
)
from triagewright.contracts.telemetry import SpanKind
from triagewright.contracts.tickets import Ticket
from triagewright.contracts.tools import KBHit, ToolName
from triagewright.contracts.world import EscalationReason, Order
from triagewright.pacing import PacingRefusedError


@dataclass
class RunState:
    """The request's fields, time budget, and guardrail screens."""

    ticket: Ticket  # PII-redacted
    request_id: str
    model: ModelClient
    gateway: ToolGateway
    guardrails: Guardrails
    executor: ActionExecutor
    telemetry: RequestTelemetry
    models: Mapping[str, str]
    config: LoopConfig
    clock: Clock
    sleep: Sleep
    started: float
    injection: bool = False
    ticket_flagged: bool = False  # the ticket itself carries a suspected injection
    flagged_before_review: bool = False  # code's own suspicion, before the reviewer's screen
    triage: TriageResult | None = None
    answer: AnswerDraft | None = None
    action_summary: str = ""
    listed_orders: list[Order] | None = None  # the last list_orders result
    listed_all: bool = False  # whether it was every order (a full page may have more)
    facts: list[str] = field(default_factory=list)
    kb_hits: dict[str, KBHit] = field(default_factory=dict)
    kb_searches: set[str] = field(default_factory=set)  # every search_kb, see ``searched``
    deferred: list[ProposedAction] = field(default_factory=list)
    irreversible_requested: bool = False  # any irreversible proposal, sent or not
    outcomes: list[ActionOutcome] = field(default_factory=list)
    tool_failures: list[ToolName] = field(default_factory=list)
    turns: dict[str, int] = field(default_factory=dict)
    paced_s: float = 0.0  # rate-limit waits so far; off the request clock

    # ---- time ------------------------------------------------------------------

    def elapsed_ms(self) -> float:
        return (self.clock() - self.started) * 1000.0

    def remaining(self, *, handoff: bool = False) -> float:
        """Seconds left. Work stops ``handoff_reserve_s`` early; the handoff may use it."""
        budget = self.config.total_timeout_s
        if not handoff:
            budget -= self.config.handoff_reserve_s
        return budget - (self.clock() - self.started - self.paced_s)

    def _pace(self, wait: float) -> None:
        """A rate-limit wait: allowed up to ``max_pacing_s`` a request, and off the clock."""
        if self.paced_s + wait > self.config.max_pacing_s:
            raise PacingRefusedError(f"pacing would exceed {self.config.max_pacing_s:g}s")
        self.paced_s += wait

    def ensure_time(self) -> float:
        left = self.remaining()
        if left <= 0:
            raise Fallback(LoopStop.TIMEOUT, EscalationReason.LOOP_LIMIT, "time budget used up")
        return left

    # ---- guardrails --------------------------------------------------------------

    def screen(self, text: str, source: ContentSource) -> bool:
        """Screen untrusted content and OR the verdict into the request's injection flag."""
        with self.telemetry.span(
            "guardrails.screen", SpanKind.GUARDRAIL, {"source": source.value}
        ) as span:
            verdict = self.guardrails.screen(text, source)
            flagged = verdict.injection_suspected or verdict.decision is GuardrailDecision.BLOCK
            span.set_attribute("injection_suspected", flagged)
        self.injection = self.injection or flagged
        return flagged

    def models_triage(self) -> TriageResult:
        """The triage result subagents see, with the code-held injection flag."""
        assert self.triage is not None
        flag = self.injection or self.triage.injection_suspected
        return self.triage.model_copy(update={"injection_suspected": flag})

    def data_prompt(self, payload: BaseModel) -> str:
        content = f"<data>{escape_data(canonical_json(payload.model_dump(mode='json')))}</data>"
        return f"{content}\n{prompts.FLAGGED_NOTE}" if self.injection else content
