from __future__ import annotations

import asyncio
import itertools
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.evals.scenarios import (
    Script,
    ScriptedResolver,
    make_case,
    respond,
    run,
    scripted_factory,
)
from triagewright.contracts.agents import FinalResponse, LoopStop
from triagewright.contracts.evals import EvalCase, ToolFault
from triagewright.contracts.telemetry import RequestMetrics
from triagewright.contracts.tickets import Ticket
from triagewright.contracts.tools import ErrorCategory, ToolError, ToolName
from triagewright.contracts.world import WorldState
from triagewright.evals import runner
from triagewright.evals.runner import CaseRecord, RunConfig, input_key, load_records, run_cases
from triagewright.pacing import CALL_BUDGET
from triagewright.testing import (
    FakeCaseRuntime,
    FakeCaseRuntimeFactory,
    sample_ticket,
    seed_as_of,
    seed_world,
)

pytestmark = pytest.mark.anyio


async def ok(rt: FakeCaseRuntime, t: Ticket) -> FinalResponse:
    return respond(t)


def cases(n: int) -> list[EvalCase]:
    return [
        make_case(f"happy-{i:03d}", ticket=sample_ticket(ticket_id=f"tkt_{i:04d}"))
        for i in range(1, n + 1)
    ]


async def run_all(
    factory: FakeCaseRuntimeFactory, selected: Sequence[EvalCase], config: RunConfig
) -> list[CaseRecord]:
    return await run_cases(
        factory, selected, seed_world=seed_world(), seed_as_of=seed_as_of(), config=config
    )


def scripts_for(selected: Sequence[EvalCase], script: Script) -> Mapping[str, Script]:
    return {c.ticket.ticket_id: script for c in selected}


async def test_each_case_gets_its_own_stack_with_world_clock_and_faults() -> None:
    seen: list[tuple[str, int, datetime, int]] = []

    class Recording(FakeCaseRuntimeFactory):
        def build(
            self,
            *,
            case_id: str,
            world: WorldState,
            as_of: datetime,
            faults: Sequence[ToolFault],
        ) -> FakeCaseRuntime:
            seen.append((case_id, len(world.customers), as_of, len(faults)))
            return super().build(case_id=case_id, world=world, as_of=as_of, faults=faults)

    custom = make_case(
        "happy-002",
        ticket=sample_ticket(ticket_id="tkt_0002"),
        world=WorldState.model_validate(
            {
                "customers": [
                    {
                        "customer_id": "cus_new99",
                        "name": "New",
                        "email": "new@example.com",
                        "created_at": "2026-01-01T00:00:00Z",
                    }
                ]
            }
        ),
        faults=[
            ToolFault(
                tool=ToolName.GET_ORDER,
                error=ToolError(category=ErrorCategory.TIMEOUT, message="t", retryable=True),
            )
        ],
    ).model_copy(update={"as_of": datetime.fromisoformat("2026-07-01T00:00:00Z")})
    selected = [cases(1)[0], custom]
    factory = Recording(lambda rt: ScriptedResolver(rt, scripts_for(selected, ok)))
    records = await run_all(factory, selected, RunConfig(mode="replay"))

    n_seed = len(seed_world().customers)
    assert seen == [
        ("happy-001", n_seed, seed_as_of(), 0),
        ("happy-002", n_seed + 1, datetime.fromisoformat("2026-07-01T00:00:00Z"), 1),
    ]
    assert [r.result.case_id for r in records] == ["happy-001", "happy-002"]
    assert all(r.result.passed for r in records)


async def test_concurrency_is_bounded_and_order_is_kept() -> None:
    active = 0
    peak = 0

    async def slow(rt: FakeCaseRuntime, t: Ticket) -> FinalResponse:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return respond(t)

    selected = cases(6)
    factory = scripted_factory(scripts_for(selected, slow))
    records = await run_all(factory, selected, RunConfig(mode="replay", concurrency=2))
    assert peak == 2
    assert [r.result.case_id for r in records] == [c.id for c in selected]


class _VirtualClock:
    """The throttle's clock and sleep: time moves only when it sleeps, so load can't skew gaps."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []
        self.late = [0.03]  # the first sleep wakes 30 ms late
        self._sleep = asyncio.sleep

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        self.sleeps.append(delay)
        self.now += delay + (self.late.pop() if self.late else 0.0)
        await self._sleep(0)


async def test_throttle_spaces_case_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = _VirtualClock()
    monkeypatch.setattr(
        runner, "time", SimpleNamespace(monotonic=clock.monotonic, perf_counter=time.perf_counter)
    )
    monkeypatch.setattr(asyncio, "sleep", clock.sleep)
    starts: list[float] = []

    async def stamp(rt: FakeCaseRuntime, t: Ticket) -> FinalResponse:
        starts.append(clock.now)
        return respond(t)

    selected = cases(3)
    factory = scripted_factory(scripts_for(selected, stamp))
    config = RunConfig(mode="replay", concurrency=3, min_interval_s=0.05)
    await run_all(factory, selected, config)
    assert clock.sleeps == pytest.approx([0.05, 0.05])  # the first start doesn't wait
    # The interval runs from the late start, not from when the sleep was due to end.
    assert [b - a for a, b in itertools.pairwise(starts)] == pytest.approx([0.08, 0.05])


async def test_results_are_appended_and_resume_skips_finished_cases(tmp_path: Path) -> None:
    results = tmp_path / "out" / "results.jsonl"
    selected = cases(3)
    factory = scripted_factory(scripts_for(selected, ok))
    await run_all(factory, selected[:2], RunConfig(mode="replay", results_path=results))
    assert len(results.read_text().splitlines()) == 2

    config = RunConfig(mode="replay", results_path=results, resume=True)
    resumed = scripted_factory(scripts_for(selected, ok))
    records = await run_all(resumed, selected, config)
    assert resumed.built == ["happy-003"]
    assert [r.result.case_id for r in records] == ["happy-001", "happy-002", "happy-003"]
    saved = {r.result.case_id for r in load_records(results).values()}
    assert saved == {"happy-001", "happy-002", "happy-003"}

    # Another mode or an edited case reruns; a fresh run truncates the file.
    live = scripted_factory(scripts_for(selected, ok))
    await run_all(live, selected[:1], RunConfig(mode="live", results_path=results, resume=True))
    assert live.built == ["happy-001"]
    edited = [selected[0], selected[1].model_copy(update={"title": "edited"})]
    again = scripted_factory(scripts_for(selected, ok))
    await run_all(again, edited, config)
    assert again.built == ["happy-002"]
    for changed in (
        RunConfig(mode="replay", results_path=results, resume=True, case_timeout_s=1.0),
        RunConfig(mode="replay", results_path=results, resume=True, stack="other-models"),
    ):
        rerun = scripted_factory(scripts_for(selected, ok))
        await run_all(rerun, selected[:1], changed)
        assert rerun.built == ["happy-001"]
    await run_all(factory, selected[:1], RunConfig(mode="replay", results_path=results))
    assert len(results.read_text().splitlines()) == 1


async def test_resume_creates_the_results_directory(tmp_path: Path) -> None:
    results = tmp_path / "fresh" / "results.jsonl"
    factory = scripted_factory(scripts_for(cases(1), ok))
    await run_all(factory, cases(1), RunConfig(mode="replay", results_path=results, resume=True))
    assert len(results.read_text().splitlines()) == 1


async def test_missing_metrics_fail_live_and_replay_cases() -> None:
    async def bare(rt: FakeCaseRuntime, t: Ticket) -> FinalResponse:
        return respond(t).model_copy(update={"metrics": None})

    record = await run(make_case(), bare)
    assert record.result.error == "response has no request metrics"

    async def live_metrics(rt: FakeCaseRuntime, t: Ticket) -> FinalResponse:
        metrics = RequestMetrics(request_id="req_00000001", ticket_id=t.ticket_id)
        return respond(t, metrics=metrics)  # estimated=False in a replay run

    record = await run(make_case(), live_metrics)
    assert record.result.error == "metrics estimated=False in a replay run"


async def test_timeout_fails_the_case_but_still_grades_safety() -> None:
    async def hangs(rt: FakeCaseRuntime, t: Ticket) -> FinalResponse:
        await asyncio.sleep(10)
        raise AssertionError("unreachable")  # pragma: no cover

    record = await run(make_case(), hangs, timeout_s=0.02)
    r = record.result
    assert r.error == "timed out after 0.02s" and not r.passed and r.status is None
    assert r.latency_ms >= 20 and not r.unsafe_action
    assert {g.grader: g.passed for g in r.grades}["status"] is False


async def test_pacing_waits_are_not_counted_against_the_case_timeout() -> None:
    async def paced(rt: FakeCaseRuntime, t: Ticket) -> FinalResponse:
        budget = CALL_BUDGET.get()
        assert budget is not None
        budget.paced(0.2)  # what llm.ratelimit reports before it sleeps
        await asyncio.sleep(0.15)
        return respond(t)

    record = await run(make_case(), paced, timeout_s=0.05)
    assert record.result.error is None


async def test_resolver_error_fails_the_case() -> None:
    async def boom(rt: FakeCaseRuntime, t: Ticket) -> FinalResponse:
        raise ValueError("agent crashed")

    record = await run(make_case(), boom)
    assert record.result.error == "resolve failed: ValueError('agent crashed')"
    assert not record.result.passed


async def test_build_error_fails_the_case() -> None:
    class Broken(FakeCaseRuntimeFactory):
        def build(
            self,
            *,
            case_id: str,
            world: WorldState,
            as_of: datetime,
            faults: Sequence[ToolFault],
        ) -> FakeCaseRuntime:
            raise RuntimeError("no database")

    [record] = await run_all(Broken(), cases(1), RunConfig(mode="replay"))
    assert record.result.error == "build failed: RuntimeError('no database')"
    assert record.result.grades == [] and not record.result.passed


async def test_mode_mismatch_fails_the_case() -> None:
    async def live(rt: FakeCaseRuntime, t: Ticket) -> FinalResponse:
        return respond(t, mode="live")

    record = await run(make_case(), live, mode="replay")
    assert record.result.error == "mode mismatch: response is live, run is replay"
    assert not record.result.passed


async def test_cost_and_latency_come_from_request_metrics() -> None:
    async def measured(rt: FakeCaseRuntime, t: Ticket) -> FinalResponse:
        metrics = RequestMetrics(
            request_id="req_00000001",
            ticket_id=t.ticket_id,
            cost_usd=0.002,
            latency_ms=1234.0,
            paced_ms=500.0,
            ttft_ms=210.0,
            estimated=True,
        )
        return respond(t, metrics=metrics).model_copy(update={"stop": LoopStop.TIMEOUT})

    r = (await run(make_case(), measured)).result
    assert (r.cost_usd, r.latency_ms, r.ttft_ms) == (0.002, 1234.0, 210.0)
    assert (r.paced_ms, r.stop) == (500.0, LoopStop.TIMEOUT)


def test_input_key_ignores_defaults_so_new_optional_fields_keep_old_keys() -> None:
    def key(case: EvalCase) -> str:
        return input_key(case, mode="replay", timeout_s=30, stack="s", judge_model=None)

    # Pinned: adding an optional field with a default must not change this key.
    base = make_case(status_any=["resolved"])
    assert key(base) == "80981ecd7dd7ce7d17f89ffed10f31acd0fc960f89c95b028f9a73aff21e6093"
    assert key(base) != key(make_case(status_any=["resolved"], min_refund_cents=1))


async def test_a_timed_out_cases_latency_leaves_out_pacing() -> None:
    async def paced_then_hangs(rt: FakeCaseRuntime, t: Ticket) -> FinalResponse:
        budget = CALL_BUDGET.get()
        assert budget is not None
        budget.paced(0.15)
        await asyncio.sleep(0.15)  # the paced wait
        await asyncio.sleep(3600)  # then the call hangs past the case timeout
        raise AssertionError("unreachable")  # pragma: no cover

    record = await run(make_case(), paced_then_hangs, timeout_s=0.05)
    r = record.result
    assert r.error == "timed out after 0.05s"
    assert 40 <= r.latency_ms < 150  # ~50 ms of work, not ~200 ms of wall time
