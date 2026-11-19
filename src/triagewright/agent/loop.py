"""The bounded model loop: one model call, and ``run_loop`` over a model's turns."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import Literal

from pydantic import BaseModel, ValidationError

from triagewright.agent import prompts
from triagewright.agent.common import Fallback, LocalTool, Refused, validation_message
from triagewright.agent.tool_calls import ToolCalls
from triagewright.contracts.agents import LoopStop
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
from triagewright.contracts.telemetry import SpanKind
from triagewright.contracts.world import EscalationReason
from triagewright.pacing import CallBudget, PacingRefusedError


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


class ModelLoop(ToolCalls):
    """The request's model calls and agentic loop, on top of its tool calls."""

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
                status = getattr(exc, "status", None)  # model errors carry the HTTP status
                http = f" (HTTP {status})" if isinstance(status, int) else ""
                span.set_error(type(exc).__name__ + http)
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
        settle: Callable[[], Awaitable[T | None]] | None = None,
    ) -> T:
        """Run one bounded agentic loop until the model returns a valid ``output``.

        The result arrives as a call to ``output_tool``, whose schema is ``output``.
        ``tool_use`` runs every requested tool and appends the results; ``end_turn``
        (or ``stop_sequence``) forces the output tool; ``max_tokens`` gets one
        continuation; ``pause_turn`` continues; ``refusal`` raises :class:`Refused`.
        Invalid output, or output ``check`` rejects with a message, gets one repair
        attempt (a ``check`` repair leaves the tools free). ``settle`` runs after all of a
        turn's tool calls and lets code return the result once they have decided it, saving
        the model the output call.
        Any bound,
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
                if settle is not None and (settled := await settle()) is not None:
                    return settled
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
