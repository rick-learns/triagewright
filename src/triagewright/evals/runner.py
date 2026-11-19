"""Runs eval cases, each on its own isolated stack from a :class:`CaseRuntimeFactory`.

Concurrency is bounded, case starts can be throttled (free-tier rate limits),
every case has a timeout, and finished cases are appended to a JSONL file so
an interrupted run can resume where it stopped.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from triagewright.contracts.agents import FinalResponse
from triagewright.contracts.common import Contract, canonical_json
from triagewright.contracts.evals import CaseResult, EvalCase
from triagewright.contracts.llm import ModelResponse, Usage
from triagewright.contracts.ports import CaseRuntimeFactory
from triagewright.contracts.world import WorldState
from triagewright.data import merge_world
from triagewright.evals.graders import Evidence, grade_deterministic
from triagewright.evals.judge import (
    JudgeClientFactory,
    grade_rubric,
    judge_failure,
    judge_recordings,
)

Mode = Literal["live", "replay", "stub"]

# none: the case has no rubric; skipped: no judge client configured;
# after_failure: deterministic graders already failed, so the judge didn't run.
JudgeStatus = Literal["none", "skipped", "after_failure", "judged"]


class CaseRecord(Contract):
    """One finished case, as appended to the results JSONL."""

    key: str  # input_key(): resume reuses a record only for identical inputs
    mode: Mode
    result: CaseResult
    judge_status: JudgeStatus = "none"
    judge_usage: Usage | None = None
    judge_cost_usd: Annotated[float, Field(ge=0)] = 0.0  # only when the provider reports it


@dataclass(frozen=True)
class JudgeConfig:
    factory: JudgeClientFactory
    model: str
    recordings_dir: Path


@dataclass(frozen=True)
class RunConfig:
    mode: Mode  # what every FinalResponse must report; a mismatch fails the case
    concurrency: int = 4
    min_interval_s: float = 0.0  # minimum gap between case starts
    case_timeout_s: float = 120.0
    stack: str = ""  # provider and models of the stack under test; part of the resume key
    results_path: Path | None = None  # JSONL of finished cases
    resume: bool = False  # keep finished cases from results_path and skip them


class _Throttle:
    def __init__(self, interval_s: float) -> None:
        self._interval = interval_s
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        if self._interval <= 0:
            return
        async with self._lock:
            delay = self._next - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            self._next = time.monotonic() + self._interval


def input_key(
    case: EvalCase, *, mode: Mode, timeout_s: float, stack: str, judge_model: str | None
) -> str:
    """Hash of everything a saved result depends on: the case and how it was run."""
    payload = {
        "case": case.model_dump(mode="json"),
        "mode": mode,
        "timeout_s": timeout_s,
        "stack": stack,
        "judge": judge_model,
    }
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def _error_result(case: EvalCase, error: str) -> CaseResult:
    return CaseResult(
        case_id=case.id,
        category=case.category,
        passed=False,
        unsafe_action=False,
        expected_escalate=case.expected.escalate,
        escalated=False,
        status=None,
        grades=[],
        error=error,
    )


async def run_case(
    factory: CaseRuntimeFactory,
    case: EvalCase,
    *,
    seed_world: WorldState,
    seed_as_of: datetime,
    mode: Mode,
    timeout_s: float,
    stack: str = "",
    judge: JudgeConfig | None = None,
) -> CaseRecord:
    """Build the case's stack, resolve its ticket, and grade the evidence. Never raises."""
    judge_model = judge.model if judge else None
    key = input_key(case, mode=mode, timeout_s=timeout_s, stack=stack, judge_model=judge_model)
    try:
        runtime = factory.build(
            case_id=case.id,
            world=merge_world(seed_world, case.world),
            as_of=case.as_of or seed_as_of,
            faults=case.faults,
        )
        before = runtime.ledger()
        first_call = len(runtime.tool_calls())
    except Exception as exc:
        failed = _error_result(case, f"build failed: {exc!r}")
        return CaseRecord(key=key, mode=mode, result=failed)

    response: FinalResponse | None = None
    error: str | None = None
    started = time.perf_counter()
    try:
        async with asyncio.timeout(timeout_s):
            response = await runtime.resolver.resolve(case.ticket)
    except TimeoutError:
        error = f"timed out after {timeout_s:g}s"
    except Exception as exc:
        error = f"resolve failed: {exc!r}"
    wall_ms = (time.perf_counter() - started) * 1000
    if response is not None and response.mode != mode:
        error = f"mode mismatch: response is {response.mode}, run is {mode}"
    elif response is not None and mode != "stub":
        if response.metrics is None:
            error = "response has no request metrics"  # cost and latency would be made up
        elif response.metrics.estimated != (mode == "replay"):
            error = f"metrics estimated={response.metrics.estimated} in a {mode} run"

    ev = Evidence(
        case=case,
        response=response,
        calls=runtime.tool_calls()[first_call:],
        before=before,
        after=runtime.ledger(),
        approvals=runtime.pending_approvals(),
    )
    grades = grade_deterministic(ev)
    judge_status: JudgeStatus = "none"
    judge_reply: ModelResponse | None = None
    if case.expected.rubric is not None:
        if error is not None or response is None or not all(g.passed for g in grades):
            judge_status = "after_failure"
        elif judge is None:
            judge_status = "skipped"
        else:
            try:
                recordings = judge_recordings(judge.recordings_dir, case.id)
                client = judge.factory(case_id=case.id, recordings=recordings)
                async with asyncio.timeout(timeout_s):  # bounds the model call
                    grade, judge_reply = await grade_rubric(
                        client, case, response, model=judge.model
                    )
            except TimeoutError:
                grade = judge_failure(f"judge timed out after {timeout_s:g}s")
            except Exception as exc:  # e.g. the factory can't open the recordings
                grade = judge_failure(f"judge setup failed: {exc!r}")
            grades.append(grade)
            judge_status = "judged"

    metrics = response.metrics if response else None
    result = CaseResult(
        case_id=case.id,
        category=case.category,
        passed=error is None and all(g.passed for g in grades),
        unsafe_action=ev.unsafe(),
        expected_escalate=case.expected.escalate,
        escalated=ev.escalation() is not None,
        status=response.status if response else None,
        grades=grades,
        cost_usd=metrics.cost_usd if metrics else 0.0,
        latency_ms=metrics.latency_ms if metrics else wall_ms,
        ttft_ms=metrics.ttft_ms if metrics else None,
        error=error,
    )
    return CaseRecord(
        key=key,
        mode=mode,
        result=result,
        judge_status=judge_status,
        judge_usage=judge_reply.usage if judge_reply else None,
        judge_cost_usd=(judge_reply.provider_cost_usd or 0.0) if judge_reply else 0.0,
    )


def load_records(path: Path) -> dict[str, CaseRecord]:
    """Finished cases from a results JSONL, by ``input_key``. Later lines win."""
    records: dict[str, CaseRecord] = {}
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = CaseRecord.model_validate(json.loads(line))
            records[record.key] = record
    return records


async def run_cases(
    factory: CaseRuntimeFactory,
    cases: Sequence[EvalCase],
    *,
    seed_world: WorldState,
    seed_as_of: datetime,
    config: RunConfig,
    judge: JudgeConfig | None = None,
) -> list[CaseRecord]:
    """Run ``cases`` and return one record per case, in case order."""
    done: dict[str, CaseRecord] = {}
    path = config.results_path
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if config.resume:
            saved = load_records(path)
            judge_model = judge.model if judge else None
            for case in cases:
                key = input_key(
                    case,
                    mode=config.mode,
                    timeout_s=config.case_timeout_s,
                    stack=config.stack,
                    judge_model=judge_model,
                )
                record = saved.get(key)
                if record is not None:
                    done[case.id] = record
        else:
            path.write_text("", encoding="utf-8")

    slots = asyncio.Semaphore(max(1, config.concurrency))
    throttle = _Throttle(config.min_interval_s)

    async def one(case: EvalCase) -> CaseRecord:
        async with slots:
            await throttle.wait()
            record = await run_case(
                factory,
                case,
                seed_world=seed_world,
                seed_as_of=seed_as_of,
                mode=config.mode,
                timeout_s=config.case_timeout_s,
                stack=config.stack,
                judge=judge,
            )
        if path is not None:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(record.model_dump_json() + "\n")
        return record

    todo = [c for c in cases if c.id not in done]
    fresh = await asyncio.gather(*(one(c) for c in todo))
    done.update({r.result.case_id: r for r in fresh})
    return [done[c.id] for c in cases]
