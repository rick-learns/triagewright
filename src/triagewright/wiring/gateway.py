"""The MCP tool gateway for an out-of-process server, held on a private event-loop thread."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from collections.abc import Awaitable, Callable, Sequence

import anyio

from triagewright.contracts.common import JsonObject
from triagewright.contracts.tools import (
    TOOL_SPECS,
    CallContext,
    ErrorCategory,
    ToolName,
    ToolResult,
    ToolSpec,
)
from triagewright.mcp_server import McpToolGateway

logger = logging.getLogger("triagewright.wiring")  # the name from before the package split


def _retrieve(future: asyncio.Future[McpToolGateway]) -> None:
    if not future.cancelled():
        future.exception()


class BackgroundMcpGateway:
    """A ``ToolGateway`` that holds its MCP session on a private event-loop thread.

    Services are built synchronously and may be called from more than one event
    loop (the CLI runs one per command), but an MCP session must stay in the
    task that opened it. So one owner task on this thread opens it, and every
    call hops onto that loop. A failed or stalled connect and a lost session
    are retryable ``transient`` results; the next call reconnects.
    """

    def __init__(
        self,
        open_gateway: Callable[[], McpToolGateway],
        *,
        connect_timeout_s: float = 10.0,
        close_timeout_s: float = 5.0,
    ) -> None:
        self._open = open_gateway
        self._connect_timeout_s = connect_timeout_s
        self._close_timeout_s = close_timeout_s
        self._lock = threading.Lock()  # guards _accepting and _pending (caller threads)
        self._accepting = True
        self._pending: set[concurrent.futures.Future[ToolResult]] = set()
        # Loop-thread state:
        self._loop = asyncio.new_event_loop()
        self._session: asyncio.Future[McpToolGateway] | None = None
        self._owner: asyncio.Task[None] | None = None  # owns _session
        self._owners: set[asyncio.Task[None]] = set()  # incl. abandoned ones still tearing down
        self._closing = asyncio.Event()
        self._thread = threading.Thread(target=self._loop.run_forever, name="mcp-gateway")
        self._thread.daemon = True
        self._thread.start()

    def list_tools(self) -> Sequence[ToolSpec]:
        return list(TOOL_SPECS.values())

    async def call(self, name: ToolName, arguments: JsonObject, ctx: CallContext) -> ToolResult:
        with self._lock:  # close() can't slip between the check and the submit
            if not self._accepting:
                return ToolResult.fail(name, ErrorCategory.TRANSIENT, "MCP gateway closed")
            future = asyncio.run_coroutine_threadsafe(
                self._call(name, dict(arguments), ctx), self._loop
            )
            self._pending.add(future)
        future.add_done_callback(self._settled)
        try:
            return await asyncio.wrap_future(future)  # cancelling the caller cancels the call
        except asyncio.CancelledError:
            task = asyncio.current_task()
            if task is not None and task.cancelling():
                raise  # the caller was cancelled
            return ToolResult.fail(name, ErrorCategory.TRANSIENT, "MCP gateway closed")

    def close(self) -> None:
        """Stop taking calls, end the session, settle in-flight calls, stop the thread.

        Idempotent. Each wait is bounded, so a stalled server can't hang shutdown.
        """
        with self._lock:
            if not self._accepting:
                return
            self._accepting = False
            pending = list(self._pending)
        shutdown = asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)
        try:
            shutdown.result(timeout=self._close_timeout_s)
        except (TimeoutError, concurrent.futures.CancelledError):
            shutdown.cancel()
        _, stuck = concurrent.futures.wait(pending, timeout=self._close_timeout_s)
        for future in stuck:
            future.cancel()
        concurrent.futures.wait(stuck, timeout=self._close_timeout_s)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=self._close_timeout_s)
        if not self._thread.is_alive():
            self._loop.close()

    def _settled(self, future: concurrent.futures.Future[ToolResult]) -> None:
        with self._lock:
            self._pending.discard(future)

    async def _call(self, name: ToolName, arguments: JsonObject, ctx: CallContext) -> ToolResult:
        try:
            session = self._connect()
            gateway = await asyncio.wait_for(asyncio.shield(session), self._connect_timeout_s)
        except TimeoutError:
            self._abandon(session)
            logger.warning("MCP connect timed out after %gs", self._connect_timeout_s)
            return ToolResult.fail(name, ErrorCategory.TRANSIENT, "MCP server unavailable")
        except Exception as exc:
            logger.warning("MCP connect failed: %s", type(exc).__name__)
            return ToolResult.fail(name, ErrorCategory.TRANSIENT, "MCP server unavailable")
        return await gateway.call(name, arguments, ctx)

    def _connect(self) -> asyncio.Future[McpToolGateway]:
        if self._closing.is_set():
            raise ConnectionError("gateway closing")
        session = self._session
        if session is None or (session.done() and (session.exception() or session.result().lost)):
            session = self._session = self._loop.create_future()
            session.add_done_callback(_retrieve)  # a failure nobody awaited isn't logged as lost
            self._owner = owner = self._loop.create_task(self._own(session))
            self._owners.add(owner)
            owner.add_done_callback(self._owners.discard)
        return session

    def _abandon(self, session: asyncio.Future[McpToolGateway]) -> None:
        """A stalled connect: fail it, then cancel it so the next call starts over."""
        if self._session is session and not session.done():
            self._session = None
            self._cancel_owner(session, "MCP connect abandoned")

    def _cancel_owner(self, session: asyncio.Future[McpToolGateway], why: str) -> None:
        # Settle first: an owner cancelled before it starts never reaches its finally.
        if not session.done():
            session.set_exception(ConnectionError(why))
        if self._owner is not None:
            self._owner.cancel()

    async def _own(self, ready: asyncio.Future[McpToolGateway]) -> None:
        try:
            async with self._open() as gateway:
                if ready.done():  # abandoned while connecting: close it at once
                    return
                ready.set_result(gateway)
                async with anyio.create_task_group() as tg:

                    async def until(wait: Callable[[], Awaitable[object]]) -> None:
                        await wait()
                        tg.cancel_scope.cancel()

                    tg.start_soon(until, gateway.wait_lost)
                    tg.start_soon(until, self._closing.wait)
        except Exception as exc:
            if not ready.done():
                ready.set_exception(exc)
            else:
                logger.warning("MCP session closed with %s", type(exc).__name__)
        finally:
            if not ready.done():  # cancelled mid-connect
                ready.set_exception(ConnectionError("MCP connect cancelled"))

    async def _shutdown(self) -> None:
        self._closing.set()
        if self._session is not None and not self._session.done():
            self._cancel_owner(self._session, "MCP gateway closed")  # never sees _closing
        # Every owner, including abandoned ones, exits its session (and reaps a stdio child).
        await asyncio.gather(*self._owners, return_exceptions=True)
