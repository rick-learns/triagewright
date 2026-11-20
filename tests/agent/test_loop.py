"""The coordinator loop: every stop_reason, every bound, retries, and invalid output."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from tests.agent.helpers import (
    FakeClock,
    TimedModel,
    call,
    finalize,
    request_text,
    review,
    stack,
    submit,
    text,
    tool_results,
    triage,
)
from triagewright.agent import prompts
from triagewright.contracts.agents import LoopConfig, LoopStop, ResolutionStatus
from triagewright.contracts.common import JsonObject
from triagewright.contracts.evals import ToolFault
from triagewright.contracts.llm import (
    ModelRequest,
    ModelResponse,
    Provider,
    StopReason,
    TextBlock,
    ToolChoice,
)
from triagewright.contracts.telemetry import ModelPrice
from triagewright.contracts.tools import ErrorCategory, ToolError, ToolName
from triagewright.llm.errors import ModelUnavailableError
from triagewright.observability import OTelTelemetryFactory, to_trace_event
from triagewright.observability.pricing import PriceTable
from triagewright.pacing import CALL_BUDGET
from triagewright.testing import ScriptedModelClient, model_response, sample_ticket

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


ORDER: JsonObject = {"order_id": "ord_a1002"}


def transient(times: int, retry_after_ms: int | None = None) -> ToolFault:
    error = ToolError(
        category=ErrorCategory.TRANSIENT,
        message="backend busy",
        retryable=True,
        retry_after_ms=retry_after_ms,
    )
    return ToolFault(tool=ToolName.GET_ORDER, error=error, times=times)


# ---- stop reasons ----------------------------------------------------------------


async def test_tool_use_runs_every_block_and_appends_results() -> None:
    both = model_response(
        *call("get_order", ORDER).content,
        *call("get_account", {"customer_id": "cus_ava01"}).content,
    )
    s = stack([triage(), both, finalize(), review()])
    response = await s.resolve(sample_ticket())
    assert response.status is ResolutionStatus.RESOLVED
    assert response.stop is LoopStop.COMPLETED
    assert response.turns == 2
    assert [c.tool for c in s.calls()] == [ToolName.GET_ORDER, ToolName.GET_ACCOUNT]
    assert {c.caller for c in s.calls()} == {"coordinator"}
    second = s.requests("coordinator")[1]
    results = tool_results(second)
    assert [r.is_error for r in results] == [False, False]
    assert "1Z1002003" in results[0].content  # the order's tracking number
    assert second.messages[-2].role == "assistant"


@pytest.mark.parametrize("stop", [StopReason.END_TURN, StopReason.STOP_SEQUENCE])
async def test_end_turn_forces_finalize(stop: StopReason) -> None:
    s = stack([triage(), text("I think we're done.", stop), finalize(), review()])
    response = await s.resolve(sample_ticket())
    assert response.status is ResolutionStatus.RESOLVED
    forced = s.requests("coordinator")[1]
    assert forced.tool_choice == ToolChoice(type="tool", name="finalize")
    assert forced.messages[-1].content == [TextBlock(text=prompts.force_output("finalize"))]


async def test_max_tokens_gets_one_continuation() -> None:
    s = stack([triage(), text("Your order", StopReason.MAX_TOKENS), finalize(), review()])
    response = await s.resolve(sample_ticket())
    assert response.status is ResolutionStatus.RESOLVED
    retry = s.requests("coordinator")[1]
    assert retry.messages[-2].content == [TextBlock(text="Your order")]
    assert retry.messages[-1].content == [TextBlock(text=prompts.CONTINUE_AFTER_MAX_TOKENS)]


async def test_the_continuation_gets_twice_the_token_limit() -> None:
    # A reasoning model's thinking counts against max_tokens: the same limit would cut it again.
    s = stack([triage(), text("Your order", StopReason.MAX_TOKENS), finalize(), review()])
    await s.resolve(sample_ticket())
    first, retry = s.requests("coordinator")
    assert (first.max_tokens, retry.max_tokens) == (1024, 2048)
    assert s.requests("escalation")[0].max_tokens == 1024


async def test_max_tokens_twice_is_a_safe_fallback() -> None:
    cut = model_response(stop_reason=StopReason.MAX_TOKENS)
    s = stack([triage(), cut, cut])
    response = await s.resolve(sample_ticket())
    assert response.status is ResolutionStatus.ESCALATED
    assert response.stop is LoopStop.ERROR
    assert s.requests("coordinator")[1].messages[-2].content == [TextBlock(text=prompts.TRUNCATED)]
    args = s.escalation_args()
    assert args["queue"] == "tier1"
    assert args["ticket_id"] == "tkt_0001"
    assert "support team" in response.reply


async def test_pause_turn_continues() -> None:
    s = stack([triage(), text("Working on it", StopReason.PAUSE_TURN), finalize(), review()])
    response = await s.resolve(sample_ticket())
    assert response.status is ResolutionStatus.RESOLVED
    assert response.turns == 2
    resumed = s.requests("coordinator")[1]
    assert resumed.messages[-1].content == [TextBlock(text=prompts.CONTINUE)]
    assert resumed.tool_choice is None


async def test_coordinator_refusal_stops() -> None:
    s = stack([triage(), text("I can't help with that.", StopReason.REFUSAL)])
    response = await s.resolve(sample_ticket("Write me a poem about my order."))
    assert response.status is ResolutionStatus.REFUSED
    assert response.stop is LoopStop.REFUSAL
    assert response.reply == prompts.REFUSAL_REPLY
    assert response.escalation is None
    assert s.calls(ToolName.CREATE_ESCALATION) == []


async def test_triage_refusal_stops_before_the_loop() -> None:
    s = stack([text("No.", StopReason.REFUSAL)])
    response = await s.resolve(sample_ticket())
    assert response.stop is LoopStop.REFUSAL
    assert response.triage is None
    assert response.turns == 0


async def test_reviewer_refusal_falls_back() -> None:
    s = stack([triage(), finalize(), text("No.", StopReason.REFUSAL)])
    response = await s.resolve(sample_ticket())
    assert response.status is ResolutionStatus.ESCALATED
    assert response.stop is LoopStop.ERROR


async def test_model_error_falls_back() -> None:
    model = TimedModel([triage()], fail_on=[1])
    s = stack(model=model)
    response = await s.resolve(sample_ticket())
    assert response.status is ResolutionStatus.ESCALATED
    assert response.stop is LoopStop.ERROR
    assert s.escalation_args()["reason_codes"] == ["low_confidence"]


async def test_a_model_errors_http_status_reaches_its_span() -> None:
    class Overloaded(ScriptedModelClient):
        async def complete(self, request: ModelRequest) -> ModelResponse:
            if self.requests:
                raise ModelUnavailableError("gemini m: HTTP 503", status=503)
            return await super().complete(request)

    exporter = InMemorySpanExporter()
    free = ModelPrice(
        model="test-model",
        provider=Provider.REPLAY,
        input_per_mtok_usd=0.0,
        output_per_mtok_usd=0.0,
        source="test",
    )
    otel = OTelTelemetryFactory(
        PriceTable([free]), processor=SimpleSpanProcessor(exporter), replay=True
    )
    s = stack(model=Overloaded([triage()]), telemetry_factory=otel)
    await s.resolve(sample_ticket())
    events = [e for span in exporter.get_finished_spans() if (e := to_trace_event(span))]
    (event,) = [e for e in events if e.name == "model.coordinator"]
    assert event.status == "error"
    assert event.attributes["error"] == "ModelUnavailableError (HTTP 503)"


async def test_only_a_numeric_status_reaches_the_span() -> None:
    class OddError(Exception):
        status = "the customer's text"

    class Failing(ScriptedModelClient):
        async def complete(self, request: ModelRequest) -> ModelResponse:
            if self.requests:
                raise OddError
            return await super().complete(request)

    s = stack(model=Failing([triage()]))
    await s.resolve(sample_ticket())
    assert s.telemetry.last is not None
    assert dict(s.telemetry.last.span_errors)["model.coordinator"] == "OddError"


# ---- bounds ------------------------------------------------------------------------


async def test_max_turns_escalates_with_a_handoff_summary() -> None:
    s = stack(
        [triage(), call("get_order", ORDER), call("get_order", ORDER)],
        config=LoopConfig(max_turns=2, retry_backoff_s=0),
    )
    response = await s.resolve(sample_ticket())
    assert response.status is ResolutionStatus.ESCALATED
    assert response.stop is LoopStop.MAX_TURNS
    assert response.turns == 2
    args = s.escalation_args()
    assert (args["queue"], args["reason_codes"]) == ("tier1", ["loop_limit"])
    summary = str(args["summary"])
    assert "Customer goal: Find out where the order is" in summary
    assert "ord_a1002" in summary  # facts checked
    assert "Why escalated: coordinator hit the turn limit" in summary
    assert response.escalation is not None and response.escalation.handoff_summary == summary


async def test_turn_timeout_escalates() -> None:
    model = TimedModel([triage(), finalize(), review()], hang_on=[1])
    s = stack(model=model, config=LoopConfig(turn_timeout_s=0.02))
    response = await s.resolve(sample_ticket())
    assert response.status is ResolutionStatus.ESCALATED
    assert response.stop is LoopStop.TIMEOUT
    assert s.escalation_args()["reason_codes"] == ["loop_limit"]


async def test_total_timeout_keeps_the_handoff_reserve() -> None:
    clock = FakeClock()
    model = TimedModel(
        [triage(), call("get_order", ORDER), call("get_order", ORDER)], clock=clock, step_s=9
    )
    config = LoopConfig(total_timeout_s=30, handoff_reserve_s=3, turn_timeout_s=20)
    s = stack(model=model, clock=clock, config=config)
    response = await s.resolve(sample_ticket())
    # Work stops at 27 s; the handoff still runs in the reserved 3 s.
    assert response.stop is LoopStop.TIMEOUT
    assert model.calls == 3
    assert len(s.calls(ToolName.GET_ORDER)) == 1
    assert s.escalation_args()["queue"] == "tier1"
    assert "support team" in response.reply


async def test_handoff_gets_one_attempt_even_after_the_budget() -> None:
    clock = FakeClock()
    model = TimedModel([triage()], clock=clock, step_s=40)
    s = stack(model=model, clock=clock, config=LoopConfig(total_timeout_s=30))
    response = await s.resolve(sample_ticket())
    assert response.stop is LoopStop.TIMEOUT
    assert len(s.calls(ToolName.CREATE_ESCALATION)) == 1


class PacedModel(TimedModel):
    """Waits ``pace_s`` for a rate-limit slot before each call, as ``llm.ratelimit`` does:
    on the fake clock if given, else for real (most of the reported wait)."""

    def __init__(self, responses: list[ModelResponse], pace_s: float, **kw: Any) -> None:
        super().__init__(responses, **kw)
        self.pace_s = pace_s

    async def complete(self, request: ModelRequest) -> ModelResponse:
        budget = CALL_BUDGET.get()
        assert budget is not None
        budget.paced(self.pace_s)
        if self.clock is not None:
            self.clock.now += self.pace_s
        else:
            await asyncio.sleep(self.pace_s * 0.75)
        return await super().complete(request)


async def test_pacing_waits_are_not_counted_against_the_turn_timeout() -> None:
    model = PacedModel([triage(), finalize(), review()], pace_s=0.2)
    s = stack(model=model, config=LoopConfig(turn_timeout_s=0.05))
    response = await s.resolve(sample_ticket())
    assert response.status is ResolutionStatus.RESOLVED and model.calls == 3


async def test_pacing_waits_are_not_counted_against_the_total_timeout() -> None:
    clock = FakeClock()
    model = PacedModel([triage(), finalize(), review()], pace_s=20, clock=clock)
    s = stack(model=model, clock=clock, config=LoopConfig(total_timeout_s=30))
    response = await s.resolve(sample_ticket())
    assert response.status is ResolutionStatus.RESOLVED and clock.now == 60


async def test_a_failed_calls_pacing_is_still_reported() -> None:
    clock = FakeClock()
    model = PacedModel([triage(), finalize(), review()], pace_s=2, clock=clock, fail_on=[1])
    s = stack(model=model, clock=clock)
    response = await s.resolve(sample_ticket())
    assert response.status is ResolutionStatus.ESCALATED
    assert s.telemetry.last is not None and s.telemetry.last.paced_ms == 4000  # both calls


async def test_pacing_beyond_its_budget_escalates() -> None:
    clock = FakeClock()
    model = PacedModel([triage(), finalize(), review()], pace_s=20, clock=clock)
    s = stack(model=model, clock=clock, config=LoopConfig(max_pacing_s=50))
    response = await s.resolve(sample_ticket())
    assert response.status is ResolutionStatus.ESCALATED
    assert response.stop is LoopStop.TIMEOUT
    assert model.calls == 2  # the reviewer's call, at 60s of pacing, was never sent
    assert s.escalation_args()["reason_codes"] == ["loop_limit"]


# ---- retries -----------------------------------------------------------------------


async def test_retryable_errors_are_retried_with_backoff() -> None:
    s = stack(
        [triage(), call("get_order", ORDER), finalize(), review()],
        faults=[transient(times=2)],
        config=LoopConfig(retry_backoff_s=0.5),
    )
    response = await s.resolve(sample_ticket())
    assert response.status is ResolutionStatus.RESOLVED
    assert len(s.calls(ToolName.GET_ORDER)) == 3  # fault times=2 succeeds on the third call
    assert s.sleeps == [0.5, 1.0]
    assert tool_results(s.requests("coordinator")[1])[0].is_error is False


async def test_retry_after_ms_sets_the_backoff() -> None:
    s = stack(
        [triage(), call("get_order", ORDER), finalize(), review()],
        faults=[transient(times=1, retry_after_ms=250)],
    )
    await s.resolve(sample_ticket())
    assert s.sleeps == [0.25]


async def test_persistent_tool_failure_escalates_tier1() -> None:
    s = stack(
        [triage(), call("get_order", ORDER), finalize("Sorry, I couldn't check."), review()],
        faults=[transient(times=5)],
        config=LoopConfig(retry_backoff_s=0),
    )
    response = await s.resolve(sample_ticket())
    assert len(s.calls(ToolName.GET_ORDER)) == 3  # 1 + tool_retries
    assert s.sleeps == []
    assert response.status is ResolutionStatus.ESCALATED
    args = s.escalation_args()
    assert (args["queue"], args["reason_codes"]) == ("tier1", ["tool_failure"])
    assert "get_order" in str(args["summary"])


async def test_a_tool_still_failing_after_the_retries_tells_the_model_to_stop() -> None:
    # toolfail-008 live: the model read "retryable" and searched a rate-limited KB three ways.
    s = stack(
        [triage(), call("get_order", ORDER), finalize("Sorry, I couldn't check."), review()],
        faults=[transient(times=5)],
        config=LoopConfig(retry_backoff_s=0),
    )
    await s.resolve(sample_ticket())
    seen = tool_results(s.requests("coordinator")[1])[0]
    assert seen.is_error
    assert seen.content.endswith(prompts.TOOL_DOWN.format(tool="get_order"))


async def test_the_coordinator_hears_a_subagent_s_tool_is_down() -> None:
    # toolfail-008 live: after the answer subagent's search failed through its retries, the
    # coordinator, which sees only the draft, searched again itself.
    error = ToolError(category=ErrorCategory.RATE_LIMITED, message="slow down", retryable=True)
    down = ToolFault(tool=ToolName.SEARCH_KB, error=error, times=10)
    draft: JsonObject = {"answer": "I couldn't check.", "grounded": False, "confidence": 0.2}
    s = stack(
        [
            triage(),
            call("answer"),
            call("search_kb", {"query": "return window"}),
            submit("answer", draft),
            finalize("Sorry, I couldn't check."),
            review(),
        ],
        faults=[down],
        config=LoopConfig(retry_backoff_s=0),
    )
    await s.resolve(sample_ticket("How long do I have to return an item?"))
    (answered,) = tool_results(s.requests("coordinator")[1])
    assert answered.content.endswith(prompts.TOOL_DOWN.format(tool="search_kb"))
    assert len(s.calls(ToolName.SEARCH_KB)) == 3


async def test_a_non_retryable_error_gets_no_stop_note() -> None:
    s = stack(
        [
            triage(),
            call("get_order", {"order_id": "ord_zzzz"}),
            finalize("I couldn't find that order. Could you check the id?", "needs_info"),
            review(),
        ]
    )
    await s.resolve(sample_ticket("Where is ord_zzzz?"))
    seen = tool_results(s.requests("coordinator")[1])[0]
    assert seen.is_error and "retried" not in seen.content


async def test_fallback_after_a_tool_failure_keeps_the_reason() -> None:
    s = stack(
        [triage(), call("get_order", ORDER)],
        faults=[transient(times=5)],
        config=LoopConfig(max_turns=1, retry_backoff_s=0),
    )
    response = await s.resolve(sample_ticket())
    assert response.stop is LoopStop.MAX_TURNS
    assert s.escalation_args()["reason_codes"] == ["loop_limit", "tool_failure"]


async def test_backoff_longer_than_the_budget_stops_retrying() -> None:
    s = stack(
        [triage(), call("get_order", ORDER), finalize(), review()],
        faults=[transient(times=5, retry_after_ms=60_000)],
    )
    response = await s.resolve(sample_ticket())
    assert len(s.calls(ToolName.GET_ORDER)) == 1
    assert response.status is ResolutionStatus.ESCALATED


async def test_non_retryable_errors_are_not_retried() -> None:
    s = stack(
        [
            triage(),
            call("get_order", {"order_id": "ord_zzzz"}),
            finalize("I couldn't find that order. Could you check the id?", "needs_info"),
            review(),
        ]
    )
    response = await s.resolve(sample_ticket("Where is ord_zzzz?"))
    assert len(s.calls(ToolName.GET_ORDER)) == 1
    assert response.status is ResolutionStatus.NEEDS_INFO
    assert response.escalation is None


# ---- structured output -------------------------------------------------------------


async def test_finalize_mixed_with_other_tools_waits_for_their_results() -> None:
    mixed = model_response(
        *call("get_order", {"order_id": "ord_zzzz"}).content,
        *finalize().content,
    )
    s = stack(
        [
            triage(),
            mixed,
            finalize("I couldn't find that order. Could you check the id?", "needs_info"),
            review(),
        ]
    )
    response = await s.resolve(sample_ticket("Where is ord_zzzz?"))
    assert response.status is ResolutionStatus.NEEDS_INFO
    retry = s.requests("coordinator")[1]
    found, rejected = tool_results(retry)
    assert found.is_error and not found.content.startswith("Not accepted")
    assert rejected.is_error and rejected.content == prompts.output_alone("finalize")
    assert retry.tool_choice is None  # free to look up something else first


async def test_invalid_subagent_output_gets_one_repair() -> None:
    bad = submit("triage", {"category": "nope"})
    s = stack([bad, triage(), finalize(), review()])
    response = await s.resolve(sample_ticket())
    assert response.status is ResolutionStatus.RESOLVED
    repair = s.requests("triage")[1]
    (result,) = tool_results(repair)
    assert result.is_error and result.content.startswith("Invalid arguments.")
    assert repair.tool_choice == ToolChoice(type="tool", name="submit_triage")


async def test_invalid_output_twice_falls_back() -> None:
    bad = submit("triage", {"category": "nope"})
    s = stack([bad, bad])
    response = await s.resolve(sample_ticket())
    assert response.status is ResolutionStatus.ESCALATED
    assert response.stop is LoopStop.ERROR
    assert response.triage is None


async def test_no_output_when_forced_twice_falls_back() -> None:
    s = stack([text("hmm"), text("hmm")])
    response = await s.resolve(sample_ticket())
    assert response.stop is LoopStop.ERROR
    assert s.requests("triage")[0].tool_choice == ToolChoice(type="tool", name="submit_triage")


async def test_a_rejected_output_after_a_forced_turn_frees_the_tools() -> None:
    mixed = model_response(*call("get_order", ORDER).content, *finalize().content)
    s = stack([triage(), text("Done."), mixed, finalize(), review()])
    response = await s.resolve(sample_ticket())
    assert response.status is ResolutionStatus.RESOLVED
    forced, freed = s.requests("coordinator")[1:3]
    assert forced.tool_choice == ToolChoice(type="tool", name="finalize")
    assert freed.tool_choice is None


async def test_invalid_finalize_is_repaired() -> None:
    bad = call("finalize", {"status": "escalated", "reply": "Escalated!"})
    s = stack([triage(), bad, finalize(), review()])
    response = await s.resolve(sample_ticket())
    assert response.status is ResolutionStatus.RESOLVED
    assert response.reply == "Your order is on its way."


async def test_unknown_tool_is_an_error_result() -> None:
    s = stack([triage(), call("delete_everything"), finalize(), review()])
    await s.resolve(sample_ticket())
    (result,) = tool_results(s.requests("coordinator")[1])
    assert result.is_error and "Unknown tool" in result.content
    assert "delete_everything" in request_text(s.requests("coordinator")[1])
