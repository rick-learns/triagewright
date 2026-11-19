"""Keeping rate-limit waits off deadlines.

Client-side pacing (``llm.ratelimit``) is queueing, not work, so it must not
eat the agent's turn and request timeouts or the eval runner's case timeout.
Whoever sets a deadline installs a ``CallBudget`` (``CALL_BUDGET``). The
limiter reports each wait to it before sleeping, and the budget moves its
``asyncio.Timeout`` back by that much and tells the enclosing budget to do the
same. A budget may refuse a wait by raising ``PacingRefusedError``. Retries read the
deadline from the budget so they never sleep past it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar


class PacingRefusedError(Exception):
    """A budget refused to wait any longer for a rate-limit slot."""


class CallBudget:
    """Keeps pacing waits out of an ``asyncio.Timeout``: each wait moves it back by as much.

    ``on_wait`` sees every wait first and may raise ``PacingRefusedError`` to cap the total.
    Waits are passed on to the budget that was current when this one was made.
    """

    def __init__(
        self, timeout: asyncio.Timeout, *, on_wait: Callable[[float], None] | None = None
    ) -> None:
        self._timeout = timeout
        self._on_wait = on_wait
        self._outer = CALL_BUDGET.get()

    def paced(self, wait: float) -> None:
        """Called on the waiting task's loop before (and after a late wake, for the excess)."""
        if self._on_wait is not None:
            self._on_wait(wait)
        if self._outer is not None:
            self._outer.paced(wait)
        when = self._timeout.when()
        if when is not None:
            self._timeout.reschedule(when + wait)

    def deadline(self) -> float | None:
        """The loop time by which the call must finish (the earliest in the chain), or ``None``."""
        times = [self._timeout.when(), self._outer.deadline() if self._outer else None]
        return min((t for t in times if t is not None), default=None)

    @contextmanager
    def installed(self) -> Iterator[CallBudget]:
        token = CALL_BUDGET.set(self)
        try:
            yield self
        finally:
            CALL_BUDGET.reset(token)


CALL_BUDGET: ContextVar[CallBudget | None] = ContextVar("tw_call_budget", default=None)
