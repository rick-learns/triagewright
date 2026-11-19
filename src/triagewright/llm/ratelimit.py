"""Client-side request pacing for free-tier quotas.

Gemini's free tier allows a fixed number of requests per minute per model and
project (15 for Flash-Lite), and OpenRouter's ``:free`` models a fixed number
per minute per key. One ticket makes several calls in a row, so a burst of
tickets trips the quota and the 429s turn into handoffs. Pacing below the
quota on our side avoids that.

``RateLimiter`` is a sliding window: at most ``per_minute`` request starts in
any ``window_s`` span. A single ticket's calls go through at once; only the
excess waits. Reservations are taken under a thread lock and waited out with
the caller's ``sleep``, so one limiter serves clients on any event loop or
thread in the process (the CLI, the API, and eval cases all share it).
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable

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
        self._starts: deque[float] = deque()  # reserved start times, ascending

    def reserve(self) -> float:
        """Reserve the next start slot. Returns how long to wait for it (0 if none)."""
        with self._lock:
            now = self._clock()
            while self._starts and self._starts[0] <= now - self._window_s:
                self._starts.popleft()
            slot = now
            if len(self._starts) >= self.per_minute:
                slot = max(slot, self._starts[-self.per_minute] + self._window_s)
            if self._starts:
                slot = max(slot, self._starts[-1])  # keep the queue in order
            self._starts.append(slot)
            return slot - now

    async def acquire(self) -> float:
        """Wait for a slot. Returns the seconds waited."""
        wait = self.reserve()
        if wait > 0:
            await self._sleep(wait)
        return wait


class RateLimits:
    """One limiter per model, shared by every client built with it."""

    def __init__(self, per_minute: int | None) -> None:
        self.per_minute = per_minute
        self._lock = threading.Lock()
        self._limiters: dict[str, RateLimiter] = {}

    def for_model(self, model: str) -> RateLimiter | None:
        if self.per_minute is None:
            return None
        with self._lock:
            if model not in self._limiters:
                self._limiters[model] = RateLimiter(self.per_minute)
            return self._limiters[model]
