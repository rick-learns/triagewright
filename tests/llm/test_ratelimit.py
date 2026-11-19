from __future__ import annotations

import asyncio
import itertools
import threading

import pytest

from triagewright.llm.ratelimit import RateLimiter, RateLimits
from triagewright.pacing import CALL_BUDGET, CallBudget, PacingRefusedError

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class Clock:
    """A fake monotonic clock; its ``sleep`` advances it (plus ``late`` seconds of oversleep)."""

    def __init__(self, late: float = 0.0) -> None:
        self.now = 1000.0
        self.late = late
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds + self.late


def limiter(per_minute: int, clock: Clock) -> RateLimiter:
    return RateLimiter(per_minute, window_s=60, clock=clock, sleep=clock.sleep)


def test_a_burst_under_the_quota_goes_through_at_once() -> None:
    lim = limiter(3, Clock())
    assert [lim.try_admit() for _ in range(3)] == [0.0, 0.0, 0.0]


def test_excess_waits_for_the_window_to_slide() -> None:
    clock = Clock()
    lim = limiter(2, clock)
    assert [lim.try_admit() for _ in range(2)] == [0.0, 0.0]
    clock.now += 10
    assert lim.try_admit() == 50.0  # the first admission leaves the window at t+60
    assert lim.try_admit() == 50.0  # asking again charges nothing
    clock.now += 50
    assert [lim.try_admit() for _ in range(3)] == [0.0, 0.0, 60.0]


def test_a_caller_cancelled_while_waiting_costs_no_slot() -> None:
    clock = Clock()
    lim = limiter(1, clock)

    async def cancelled_sleep(seconds: float) -> None:
        raise asyncio.CancelledError

    lim._sleep = cancelled_sleep
    assert asyncio.run(lim.acquire()) == 0.0
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(lim.acquire())
    clock.now += 60
    assert lim.try_admit() == 0.0  # the cancelled caller left nothing booked


def test_admissions_never_exceed_the_quota_even_when_waking_late() -> None:
    clock = Clock(late=4.0)
    lim = limiter(1, clock)
    admitted = []
    for _ in range(4):
        asyncio.run(lim.acquire())
        admitted.append(clock.now)
    gaps = [b - a for a, b in itertools.pairwise(admitted)]
    assert all(gap >= 60 for gap in gaps), gaps


def test_no_window_ever_holds_more_than_the_quota() -> None:
    clock = Clock()
    lim = limiter(5, clock)
    admitted = []
    for step in [0, 1, 0, 3, 0, 0, 7, 2, 0, 0, 0, 30, 0, 0, 1, 0, 0, 0, 0, 90, 0, 0]:
        clock.now += step
        asyncio.run(lim.acquire())
        admitted.append(clock.now)
    assert admitted == sorted(admitted)
    for i, first in enumerate(admitted):
        assert sum(first <= t < first + 60 for t in admitted[i:]) <= 5


def test_threads_share_one_limiter() -> None:
    lim = limiter(10, Clock())
    waits: list[float] = []

    def take() -> None:
        for _ in range(5):
            waits.append(lim.try_admit())

    threads = [threading.Thread(target=take) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(waits) == [0.0] * 10 + [60.0] * 10


def test_limits_are_per_model_optional_and_shared_per_process() -> None:
    limits = RateLimits(15)
    a = limits.for_model("gemini-3.5-flash-lite")
    assert a is limits.for_model("gemini-3.5-flash-lite") and a is not None
    assert a.per_minute == 15
    assert limits.for_model("gemini-3-flash-preview") is not a
    assert RateLimits(None).for_model("gemini-3.5-flash-lite") is None
    assert RateLimits.shared(15, quota="q") is RateLimits.shared(15, quota="q")
    assert RateLimits.shared(15, quota="q") is not RateLimits.shared(20, quota="q")
    assert RateLimits.shared(15, quota="q") is not RateLimits.shared(15, quota="other")
    assert RateLimits.shared(None, quota="q").for_model("m") is None
    with pytest.raises(ValueError, match="at least 1"):
        RateLimiter(0)


async def test_pacing_waits_move_the_deadlines_back() -> None:
    lim = RateLimiter(1, window_s=0.3)
    loop = asyncio.get_running_loop()
    seen: list[float] = []
    await lim.acquire()
    async with asyncio.timeout(1.0) as outer:
        with CallBudget(outer).installed():
            async with asyncio.timeout(0.1) as inner:
                with CallBudget(inner, on_wait=seen.append).installed() as budget:
                    before = inner.when()
                    assert before is not None
                    await lim.acquire()  # waits ~0.3s, past the inner 0.1s timeout
                    assert budget.deadline() == inner.when()
    assert 0.2 < seen[0] <= 0.3 and all(late < 0.1 for late in seen[1:])  # wait, then oversleep
    after = inner.when()
    assert after is not None and after == pytest.approx(before + sum(seen))
    assert CALL_BUDGET.get() is None and loop.time() > 0


async def test_a_budget_can_refuse_to_wait() -> None:
    lim = RateLimiter(1, window_s=30)
    await lim.acquire()

    def refuse(wait: float) -> None:
        raise PacingRefusedError(f"{wait:.0f}s is too long")

    async def paced_acquire() -> None:
        async with asyncio.timeout(5) as deadline:
            with CallBudget(deadline, on_wait=refuse).installed():
                await lim.acquire()

    with pytest.raises(PacingRefusedError, match="30s is too long"):
        await paced_acquire()
    assert lim.try_admit() > 0  # refusing charged no slot either


async def test_the_deadline_is_the_earliest_in_the_chain() -> None:
    async with asyncio.timeout(1) as outer:
        with CallBudget(outer).installed():
            async with asyncio.timeout(10) as inner:
                await asyncio.sleep(0)
                assert CallBudget(inner).deadline() == outer.when()
    assert CallBudget(asyncio.timeout(None)).deadline() is None


async def test_a_lapsed_deadline_is_not_moved_back() -> None:
    # Its timeout is due or already cancelling, so crediting it would raise or revive it.
    loop = asyncio.get_running_loop()
    seen: list[float] = []
    due = asyncio.timeout_at(loop.time() - 1)
    budget = CallBudget(due, on_wait=seen.append)
    assert budget.lapsed()
    budget.paced(5)
    when = due.when()
    assert when is not None and when < loop.time() and seen == []


async def test_a_lapsed_outer_deadline_blocks_the_credit() -> None:
    loop = asyncio.get_running_loop()
    inner = asyncio.timeout_at(loop.time() + 10)
    with CallBudget(asyncio.timeout_at(loop.time() - 1)).installed():
        budget = CallBudget(inner)
    before = inner.when()
    budget.paced(5)
    assert inner.when() == before


async def test_an_expired_timeout_is_lapsed() -> None:
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0) as gone:
            await asyncio.sleep(1)
    assert CallBudget(gone).lapsed()
