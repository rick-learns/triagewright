"""Client-side request pacing for free-tier quotas.

Gemini's free tier allows a fixed number of requests per minute per model and
project (15 for Flash-Lite), and OpenRouter's ``:free`` models a fixed number
per minute per key. One ticket makes several calls in a row, so a burst of
tickets trips the quota and the 429s turn into handoffs. Pacing below the
quota on our side avoids that.

``RateLimiter`` is a sliding window over *admissions*: at most ``per_minute``
requests are admitted in any ``window_s`` span. A slot is charged only when a
caller is admitted, never while it waits, so a caller cancelled while waiting
costs nothing, and one that wakes late rechecks the window. A single ticket's
calls go through at once; only the excess waits. The window is guarded by a
thread lock and waits use the caller's ``sleep``, so one limiter serves clients
on any event loop or thread.

Quotas are per model and project (OpenRouter: per key), so a process shares one
limiter per model and quota: ``RateLimits.shared`` is that registry, keyed by
``Settings.quota_id``.

Each wait is reported to the current ``triagewright.pacing.CALL_BUDGET``
before sleeping, so it stays off the caller's deadlines.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import ClassVar

from triagewright.pacing import CALL_BUDGET

Clock = Callable[[], float]
Sleep = Callable[[float], Awaitable[None]]

WINDOW_S = 61.0  # a minute, plus a second for clock skew between us and the provider


class RateLimiter:
    def __init__(
        self,
        per_minute: int,
        *,
        window_s: float = WINDOW_S,
        clock: Clock = time.monotonic,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        if per_minute < 1:
            raise ValueError("per_minute must be at least 1")
        self.per_minute = per_minute
        self._window_s = window_s
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._admitted: deque[float] = deque()  # admission times, ascending

    def try_admit(self) -> float:
        """Admit now and return 0, or return how long until a slot may free up."""
        with self._lock:
            now = self._clock()
            # One expiry time for both the check and the wait, so they can't disagree.
            while self._admitted and self._admitted[0] + self._window_s <= now:
                self._admitted.popleft()
            if len(self._admitted) < self.per_minute:
                self._admitted.append(now)
                return 0.0
            return self._admitted[0] + self._window_s - now

    async def acquire(self) -> float:
        """Wait until admitted, reporting waits to ``CALL_BUDGET``. Returns the seconds waited."""
        started = self._clock()
        while (wait := self.try_admit()) > 0:
            budget = CALL_BUDGET.get()
            if budget is not None:
                budget.paced(wait)
            before = self._clock()
            await self._sleep(wait)
            late = self._clock() - before - wait
            if budget is not None and late > 0:
                budget.paced(late)
        return self._clock() - started


class RateLimits:
    """One limiter per model, shared by every client built with it."""

    _registry: ClassVar[dict[tuple[str, int], RateLimits]] = {}
    _registry_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, per_minute: int | None) -> None:
        self.per_minute = per_minute
        self._lock = threading.Lock()
        self._limiters: dict[str, RateLimiter] = {}

    @classmethod
    def shared(cls, per_minute: int | None, *, quota: str) -> RateLimits:
        """The process-wide limits for one quota (``Settings.quota_id``): the API, CLI, eval
        cases, and judge all draw on it, so they share one window per model. Other quotas,
        such as another project's key, get their own."""
        if per_minute is None:
            return cls(None)
        key = (quota, per_minute)
        with cls._registry_lock:
            if key not in cls._registry:
                cls._registry[key] = cls(per_minute)
            return cls._registry[key]

    def for_model(self, model: str) -> RateLimiter | None:
        if self.per_minute is None:
            return None
        with self._lock:
            if model not in self._limiters:
                self._limiters[model] = RateLimiter(self.per_minute)
            return self._limiters[model]
