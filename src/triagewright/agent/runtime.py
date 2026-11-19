"""Per-request state, the bounded model loop, and the retrying tool-call wrapper.

One :class:`RequestRun` exists per ``resolve`` call, so a coordinator can serve
requests concurrently. It holds everything code owns for the request: the
redacted ticket, the ``injection_suspected`` flag (the OR of every ``screen``),
verified facts, the knowledge-base hits citations are checked against, the
deferred irreversible proposals, the executor outcomes, and the time budget.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ValidationError

from triagewright.agent import prompts
from triagewright.contracts.agents import (
    ActionOutcome,
    AnswerDraft,
    LoopConfig,
    LoopStop,
    ProposedAction,
    TriageResult,
)
from triagewright.contracts.common import JsonObject, canonical_json
from triagewright.contracts.guardrails import ContentSource, GuardrailDecision
from triagewright.contracts.llm import (
    ContentBlock,
    Message,
    ModelRequest,
    ModelResponse,
    StopReason,
    TextBlock,
    ToolChoice,
    ToolDefinition,
    ToolResultBlock,
    ToolUseBlock,
)
from triagewright.contracts.ports import (
    ActionExecutor,
    Guardrails,
    ModelClient,
    RequestTelemetry,
    ToolGateway,
)
from triagewright.contracts.rag import Citation
from triagewright.contracts.telemetry import SpanKind
from triagewright.contracts.tickets import Ticket
from triagewright.contracts.tools import (
    TOOL_SPECS,
    CallContext,
    Caller,
    KBHit,
    KBSearchResult,
    ListOrdersInput,
    OrderList,
    ToolName,
    ToolResult,
)
from triagewright.contracts.world import EscalationReason, Order
from triagewright.pacing import CallBudget, PacingRefusedError

_HANDOFF_FLOOR_S = 1.0

type Clock = Callable[[], float]
type Sleep = Callable[[float], Awaitable[None]]
type Handler = Callable[[JsonObject], Awaitable[tuple[str, bool]]]


class Fallback(Exception):  # noqa: N818 - a control-flow signal, not an error type
    """A bound was hit or a model failed for good: hand the ticket to a human."""

    def __init__(self, stop: LoopStop, reason: EscalationReason, detail: str) -> None:
        super().__init__(detail)
        self.stop = stop
        self.reason = reason
        self.detail = detail


class Refused(Exception):  # noqa: N818 - a control-flow signal, not an error type
    """A model returned ``stop_reason=refusal``."""


@dataclass(frozen=True)
class LocalTool:
    definition: ToolDefinition
    handler: Handler


def validation_message(exc: ValidationError) -> str:
    parts = [f"{'.'.join(str(p) for p in e['loc']) or '(root)'}: {e['msg']}" for e in exc.errors()]
    return ("Invalid arguments. " + "; ".join(parts))[:1000]


def as_data(content: str, flagged: bool) -> str:
    """Wrap flagged content in delimiters with a "data, not instructions" note."""
    if not flagged:
        return content
    return f"<data>{escape_data(content)}</data>\n{prompts.FLAGGED_NOTE}"


def escape_data(content: str) -> str:
    return content.replace("</data>", "<\\/data>")


def _append(
    messages: list[Message], role: Literal["user", "assistant"], blocks: Sequence[ContentBlock]
) -> None:
    """Append blocks, merging into the last message if it has the same role."""
    if not blocks:
        return
    if messages and messages[-1].role == role:
        last = messages.pop()
        messages.append(Message(role=last.role, content=[*last.content, *blocks]))
    else:
        messages.append(Message(role=role, content=list(blocks)))


@dataclass
class RequestRun:
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
    triage: TriageResult | None = None
    answer: AnswerDraft | None = None
    action_summary: str = ""
    listed_orders: list[Order] | None = None  # the last list_orders result
    listed_all: bool = False  # whether it was every order (a full page may have more)
    facts: list[str] = field(default_factory=list)
    kb_hits: dict[str, KBHit] = field(default_factory=dict)
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

    # ---- tools -------------------------------------------------------------------

    def context(self, caller: Caller) -> CallContext:
        return CallContext(
            request_id=self.request_id,
            ticket_id=self.ticket.ticket_id,
            customer_id=self.ticket.customer_id,
            caller=caller,
        )

    async def call_tool(
        self, name: ToolName, arguments: JsonObject, caller: Caller, *, handoff: bool = False
    ) -> ToolResult:
        """Call the gateway, retrying retryable errors up to ``tool_retries`` times.

        Every attempt is a raw gateway call, so ``ExpectedOutcome.attempts`` counts them all.
        A retryable error that outlasts the retries is recorded as a tool failure.
        """
        ctx = self.context(caller)
        attempt = 0
        while True:
            attempt += 1
            left = self.remaining(handoff=handoff)
            if left <= 0:
                if not (handoff and attempt == 1):
                    raise Fallback(
                        LoopStop.TIMEOUT, EscalationReason.LOOP_LIMIT, "time budget used up"
                    )
                left = _HANDOFF_FLOOR_S  # one handoff attempt even if work overran its budget
            attrs = {"tool": name.value, "caller": caller, "attempt": attempt}
            with self.telemetry.span(f"tool.{name.value}", SpanKind.TOOL_CALL, attrs) as span:
                started = self.clock()
                try:
                    async with asyncio.timeout(left):
                        result = await self.gateway.call(name, arguments, ctx)
                except TimeoutError:
                    span.set_error("timeout")
                    raise Fallback(
                        LoopStop.TIMEOUT, EscalationReason.LOOP_LIMIT, f"{name.value} timed out"
                    ) from None
                self.telemetry.record_tool_call(result, (self.clock() - started) * 1000.0)
                if result.error is not None:
                    span.set_error(result.error.category.value)
            error = result.error
            if error is None or not error.retryable:
                return result
            if attempt > self.config.tool_retries:
                break
            if error.retry_after_ms is not None:
                delay = error.retry_after_ms / 1000.0
            else:
                delay = self.config.retry_backoff_s * 2 ** (attempt - 1)
            if delay >= self.remaining(handoff=handoff):
                break
            if delay > 0:
                await self.sleep(delay)
        self.tool_failures.append(name)
        return result

    def observe(self, result: ToolResult, arguments: JsonObject) -> str:
        """Screen a tool result, keep its facts and KB hits, and return what models see."""
        content = canonical_json(result.model_dump(mode="json"))
        hits: list[KBHit] | None = None
        if result.tool is ToolName.SEARCH_KB and result.data is not None:
            try:
                hits = KBSearchResult.model_validate(result.data).hits
            except ValidationError:
                hits = None
        if hits is not None:
            flagged = False
            for hit in hits:
                self.kb_hits.setdefault(hit.chunk_id, hit)
                text = f"{hit.title}\n{hit.text}"
                flagged = self.screen(text, ContentSource.KB_DOCUMENT) or flagged
            return as_data(content, flagged)
        flagged = self.screen(content, ContentSource.TOOL_OUTPUT)
        if result.data is not None:
            fact = f"{result.tool.value} {canonical_json(arguments)}: {canonical_json(result.data)}"
            self.facts.append(as_data(fact, flagged))
        return as_data(content, flagged)

    def _list_orders(self, query: ListOrdersInput, data: JsonObject) -> None:
        self.listed_orders = OrderList.model_validate(data).orders
        self.listed_all = len(self.listed_orders) < query.limit

    def business_tool(self, name: ToolName, caller: Caller) -> LocalTool:
        spec = TOOL_SPECS[name]

        async def handler(arguments: JsonObject) -> tuple[str, bool]:
            result = await self.call_tool(name, arguments, caller)
            if name is ToolName.LIST_ORDERS and result.data is not None:
                with contextlib.suppress(ValidationError):
                    self._list_orders(ListOrdersInput.model_validate(arguments), result.data)
            return self.observe(result, arguments), result.is_error

        definition = ToolDefinition(
            name=name.value, description=spec.description, input_schema=spec.input_schema()
        )
        return LocalTool(definition=definition, handler=handler)

    def cite(self, chunk_id: str, quote: str) -> Citation | None:
        """A citation only for a trusted hit seen in this request whose text holds the quote."""
        hit = self.kb_hits.get(chunk_id)
        quote = quote.strip()
        if hit is None or not hit.trusted or not quote or quote not in hit.text:
            return None
        return Citation(doc_id=hit.doc_id, chunk_id=hit.chunk_id, title=hit.title, quote=quote)

    def cite_all(self, candidates: Sequence[tuple[str, str]]) -> list[Citation]:
        seen: set[tuple[str, str]] = set()
        citations: list[Citation] = []
        for chunk_id, quote in candidates:
            citation = self.cite(chunk_id, quote)
            if citation is None or (citation.chunk_id, citation.quote) in seen:
                continue
            seen.add((citation.chunk_id, citation.quote))
            citations.append(citation)
        return citations

    def defer(self, action: ProposedAction) -> None:
        """Queue an irreversible proposal until after the request's last ``screen``."""
        self.irreversible_requested = True
        key = (action.tool, canonical_json(action.arguments))
        if all((a.tool, canonical_json(a.arguments)) != key for a in self.deferred):
            self.deferred.append(action)

    # ---- model -------------------------------------------------------------------

    async def call_model(
        self,
        purpose: str,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition],
        force: str | None,
    ) -> ModelResponse:
        left = self.ensure_time()
        request = ModelRequest(
            model=self.models[purpose],
            system=system,
            messages=list(messages),
            tools=list(tools),
            tool_choice=ToolChoice(type="tool", name=force) if force else None,
            max_tokens=self.config.max_tokens,
            purpose=purpose,
        )
        started_ms = self.elapsed_ms()
        paced_before = self.paced_s
        with self.telemetry.span(
            f"model.{purpose}", SpanKind.MODEL_CALL, {"model": request.model}
        ) as span:
            try:
                async with asyncio.timeout(min(self.config.turn_timeout_s, left)) as deadline:
                    with CallBudget(deadline, on_wait=self._pace).installed():
                        response = await self.model.complete(request)
            except PacingRefusedError:
                span.set_error("pacing")
                raise Fallback(
                    LoopStop.TIMEOUT,
                    EscalationReason.LOOP_LIMIT,
                    f"{purpose} model call would wait too long for a rate-limit slot",
                ) from None
            except TimeoutError:
                span.set_error("timeout")
                raise Fallback(
                    LoopStop.TIMEOUT, EscalationReason.LOOP_LIMIT, f"{purpose} model call timed out"
                ) from None
            except Exception as exc:
                span.set_error(type(exc).__name__)
                raise Fallback(
                    LoopStop.ERROR, EscalationReason.LOW_CONFIDENCE, f"{purpose} model call failed"
                ) from exc
            finally:
                paced_ms = (self.paced_s - paced_before) * 1000
                self.telemetry.record_pacing(paced_ms)
            span.set_attribute("stop_reason", response.stop_reason.value)
        self.telemetry.record_model_call(request, response, started_ms, paced_ms=paced_ms)
        return response

    async def run_loop[T: BaseModel](
        self,
        purpose: str,
        system: str,
        prompt: str,
        tools: Sequence[LocalTool],
        output: type[T],
        output_tool: str,
        output_description: str,
        check: Callable[[T], str | None] | None = None,
    ) -> T:
        """Run one bounded agentic loop until the model returns a valid ``output``.

        The result arrives as a call to ``output_tool``, whose schema is ``output``.
        ``tool_use`` runs every requested tool and appends the results; ``end_turn``
        (or ``stop_sequence``) forces the output tool; ``max_tokens`` gets one
        continuation; ``pause_turn`` continues; ``refusal`` raises :class:`Refused`.
        Invalid output, or output ``check`` rejects with a message, gets one repair
        attempt (a ``check`` repair leaves the tools free). Any bound,
        a second truncation, or a second invalid output raises :class:`Fallback`.
        ``turn_timeout_s`` bounds each model call; tool calls are bounded by the total
        budget, and writes are never cancelled midway.
        """
        definitions = [t.definition for t in tools]
        definitions.append(
            ToolDefinition(
                name=output_tool,
                description=output_description,
                input_schema=output.model_json_schema(),
            )
        )
        handlers = {t.definition.name: t.handler for t in tools}
        messages = [Message(role="user", content=[TextBlock(text=prompt)])]
        force = not tools  # with no tools to use, ask for the result straight away
        continued = False
        strikes = 0
        for turn in range(1, self.config.max_turns + 1):
            self.turns[purpose] = self.turns.get(purpose, 0) + 1
            with self.telemetry.span(
                f"{purpose}.turn", SpanKind.AGENT_TURN, {"turn": turn}
            ) as span:
                response = await self.call_model(
                    purpose, system, messages, definitions, output_tool if force else None
                )
                stop = response.stop_reason
                texts = [b for b in response.content if isinstance(b, TextBlock)]
                uses = [b for b in response.content if isinstance(b, ToolUseBlock)]
                if stop is StopReason.REFUSAL:
                    span.set_error("refusal")
                    raise Refused(purpose)
                if stop is StopReason.MAX_TOKENS:
                    if continued:
                        raise Fallback(
                            LoopStop.ERROR,
                            EscalationReason.LOW_CONFIDENCE,
                            f"{purpose} output hit the token limit twice",
                        )
                    continued = True
                    _append(messages, "assistant", texts or [TextBlock(text=prompts.TRUNCATED)])
                    _append(messages, "user", [TextBlock(text=prompts.CONTINUE_AFTER_MAX_TOKENS)])
                    continue
                if stop is StopReason.PAUSE_TURN:
                    _append(messages, "assistant", texts)
                    _append(messages, "user", [TextBlock(text=prompts.CONTINUE)])
                    continue
                if not uses:  # end_turn, stop_sequence, or a tool_use stop with no calls
                    if force:
                        strikes += 1
                        if strikes > 1:
                            raise Fallback(
                                LoopStop.ERROR,
                                EscalationReason.LOW_CONFIDENCE,
                                f"{purpose} gave no result",
                            )
                    force = True
                    _append(messages, "assistant", texts)
                    _append(messages, "user", [TextBlock(text=prompts.force_output(output_tool))])
                    continue
                _append(messages, "assistant", response.content)
                results: list[ContentBlock] = []
                value: T | None = None
                invalid = False
                mixed = len(uses) > 1
                for use in uses:
                    if use.name == output_tool and mixed:
                        # The result must not be accepted before the model sees the others.
                        content, is_error = prompts.output_alone(output_tool), True
                        force = False  # free to use the other tools first
                    elif use.name == output_tool:
                        try:
                            value = output.model_validate(use.input)
                        except ValidationError as exc:
                            invalid = force = True
                            content, is_error = validation_message(exc), True
                        else:
                            content, is_error = "Accepted.", False
                            if check is not None and (problem := check(value)) is not None:
                                value, invalid, force = None, True, False
                                content, is_error = problem, True
                    elif (handler := handlers.get(use.name)) is not None:
                        content, is_error = await handler(use.input)
                    else:
                        content, is_error = f"Unknown tool {use.name!r}.", True
                    results.append(
                        ToolResultBlock(tool_use_id=use.id, content=content, is_error=is_error)
                    )
                if value is not None:
                    return value
                _append(messages, "user", results)
                if invalid:
                    strikes += 1
                    if strikes > 1:
                        raise Fallback(
                            LoopStop.ERROR,
                            EscalationReason.LOW_CONFIDENCE,
                            f"{purpose} returned invalid output twice",
                        )
        raise Fallback(
            LoopStop.MAX_TURNS, EscalationReason.LOOP_LIMIT, f"{purpose} hit the turn limit"
        )
