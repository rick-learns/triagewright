from __future__ import annotations

import asyncio
import threading

import pytest

from triagewright.llm.ratelimit import RateLimiter, RateLimits


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_a_burst_under_the_quota_goes_through_at_once() -> None:
    limiter = RateLimiter(3, window_s=60, clock=Clock())
    assert [limiter.reserve() for _ in range(3)] == [0.0, 0.0, 0.0]


def test_excess_waits_for_the_window_to_slide() -> None:
    clock = Clock()
    limiter = RateLimiter(2, window_s=60, clock=clock)
    assert [limiter.reserve() for _ in range(2)] == [0.0, 0.0]
    clock.now += 10
    assert limiter.reserve() == 50.0  # the first start leaves the window at t+60
    assert limiter.reserve() == 50.0  # and the second, at the same instant
    assert limiter.reserve() == 110.0  # the next window
    clock.now += 200
    assert limiter.reserve() == 0.0  # old reservations expire


def test_no_window_ever_holds_more_than_the_quota() -> None:
    clock = Clock()
    limiter = RateLimiter(5, window_s=60, clock=clock)
    starts = []
    for step in [0, 1, 0, 3, 0, 0, 7, 2, 0, 0, 0, 30, 0, 0, 1, 0, 0, 0, 0, 90, 0, 0]:
        clock.now += step
        starts.append(clock.now + limiter.reserve())
    assert starts == sorted(starts)
    for i, first in enumerate(starts):
        assert sum(first <= s < first + 60 for s in starts[i:]) <= 5


def test_threads_share_one_limiter() -> None:
    clock = Clock()
    limiter = RateLimiter(10, window_s=60, clock=clock)
    waits: list[float] = []

    def take() -> None:
        for _ in range(5):
            waits.append(limiter.reserve())

    threads = [threading.Thread(target=take) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(waits) == [0.0] * 10 + [60.0] * 10


def test_acquire_sleeps_for_the_reserved_wait() -> None:
    slept: list[float] = []

    async def sleep(s: float) -> None:
        slept.append(s)

    limiter = RateLimiter(1, window_s=60, clock=Clock(), sleep=sleep)
    assert asyncio.run(limiter.acquire()) == 0.0
    assert asyncio.run(limiter.acquire()) == 60.0
    assert slept == [60.0]


def test_limits_are_per_model_and_optional() -> None:
    limits = RateLimits(15)
    a = limits.for_model("gemini-3.5-flash-lite")
    assert a is limits.for_model("gemini-3.5-flash-lite") and a is not None
    assert a.per_minute == 15
    assert limits.for_model("gemini-3-flash-preview") is not a
    assert RateLimits(None).for_model("gemini-3.5-flash-lite") is None
    with pytest.raises(ValueError, match="at least 1"):
        RateLimiter(0)
