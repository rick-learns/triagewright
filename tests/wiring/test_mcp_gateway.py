"""The background MCP gateway: one session across event loops, reconnects, close."""

from __future__ import annotations

import asyncio
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any, cast

import pytest
from mcp import Client

from tests.mcp_server.conftest import ctx
from tests.wiring.helpers import settings
from triagewright import wiring
from triagewright.contracts.tools import ErrorCategory, ToolName, ToolResult
from triagewright.mcp_server import McpToolGateway, MockSystems, ToolService, build_server
from triagewright.testing import TEST_APPROVAL_KEY, FakeRetriever, seed_as_of, seed_kb, seed_world

pytestmark = pytest.mark.anyio


def memory_gateway() -> McpToolGateway:
    service = ToolService(
        MockSystems.in_memory(seed_world()),
        as_of=seed_as_of(),
        approval_key=TEST_APPROVAL_KEY,
        retriever=FakeRetriever(seed_kb()),
    )
    return McpToolGateway(Client(build_server(service), mode="legacy"))


def test_background_gateway_serves_calls_from_separate_event_loops() -> None:
    opened: list[McpToolGateway] = []

    def open_gateway() -> McpToolGateway:
        opened.append(memory_gateway())
        return opened[-1]

    gateway = wiring.BackgroundMcpGateway(open_gateway)
    try:
        for _ in range(2):  # the CLI runs one event loop per command
            result = asyncio.run(gateway.call(ToolName.GET_ORDER, {"order_id": "ord_a1002"}, ctx()))
            assert not result.is_error, result
        assert len(opened) == 1  # one session, reused
    finally:
        gateway.close()
        gateway.close()  # idempotent
    assert opened[0].lost


def test_background_gateway_reconnects_after_a_failed_connect() -> None:
    attempts: list[int] = []

    def open_gateway() -> McpToolGateway:
        attempts.append(1)
        if len(attempts) == 1:
            raise OSError("connection refused")
        return memory_gateway()

    gateway = wiring.BackgroundMcpGateway(open_gateway)
    try:
        down = asyncio.run(gateway.call(ToolName.GET_ORDER, {"order_id": "ord_a1002"}, ctx()))
        assert down.is_error and down.error is not None
        assert down.error.category is ErrorCategory.TRANSIENT and down.error.retryable
        up = asyncio.run(gateway.call(ToolName.GET_ORDER, {"order_id": "ord_a1002"}, ctx()))
        assert not up.is_error
    finally:
        gateway.close()


class _StalledGateway:
    """Never finishes the MCP handshake."""

    lost = False

    async def __aenter__(self) -> _StalledGateway:
        await asyncio.sleep(3600)
        raise AssertionError("unreachable")

    async def __aexit__(self, *exc: object) -> None:
        pass


def test_a_stalled_connect_times_out_retries_and_does_not_hang_close() -> None:
    opens: list[int] = []

    def open_gateway() -> McpToolGateway:
        opens.append(1)
        return cast(McpToolGateway, _StalledGateway())

    gateway = wiring.BackgroundMcpGateway(open_gateway, connect_timeout_s=0.1, close_timeout_s=1)
    for _ in range(2):
        result = asyncio.run(gateway.call(ToolName.GET_ORDER, {"order_id": "ord_a1002"}, ctx()))
        assert result.error and result.error.category is ErrorCategory.TRANSIENT
    assert len(opens) == 2  # the stalled attempt was abandoned, not reused
    started = time.monotonic()
    gateway.close()
    assert time.monotonic() - started < 1


class _BlockingService(ToolService):
    def __init__(self) -> None:
        super().__init__(
            MockSystems.in_memory(seed_world()), as_of=seed_as_of(), approval_key=TEST_APPROVAL_KEY
        )
        self.entered = threading.Event()

    async def call(self, *args: Any, **kwargs: Any) -> Any:
        self.entered.set()
        await asyncio.sleep(3600)


def test_close_settles_an_in_flight_call_and_refuses_new_ones() -> None:
    service = _BlockingService()
    gateway = wiring.BackgroundMcpGateway(
        lambda: McpToolGateway(Client(build_server(service), mode="legacy")), close_timeout_s=2
    )
    results: list[Any] = []

    def caller() -> None:
        results.append(asyncio.run(gateway.call(ToolName.GET_ORDER, {"order_id": "o"}, ctx())))

    thread = threading.Thread(target=caller)
    thread.start()
    assert service.entered.wait(5)
    gateway.close()
    thread.join(5)
    (in_flight,) = results
    assert in_flight.error and in_flight.error.category is ErrorCategory.TRANSIENT
    after = asyncio.run(gateway.call(ToolName.GET_ORDER, {"order_id": "o"}, ctx()))
    assert after.error and after.error.message == "MCP gateway closed"


def _mcp_children() -> list[int]:
    me = str(os.getpid())
    found = []
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = stat.read_text().rsplit(")", 1)[1].split()
            cmdline = (stat.parent / "cmdline").read_bytes()
        except OSError:
            continue
        if fields[1] == me and b"triagewright.mcp_server" in cmdline:
            found.append(int(stat.parent.name))
    return found


@pytest.mark.integration
@pytest.mark.skipif(not Path("/proc").is_dir(), reason="needs /proc to find the child")
def test_a_killed_stdio_server_is_replaced_on_the_next_calls(tmp_path: Path) -> None:
    stdio = settings(tmp_path, mcp_transport="stdio", approval_key="k" * 32)
    gateway = wiring.BackgroundMcpGateway(lambda: McpToolGateway.from_settings(stdio))

    def get() -> Any:
        return asyncio.run(gateway.call(ToolName.GET_ORDER, {"order_id": "ord_a1002"}, ctx()))

    try:
        assert not get().is_error
        (child,) = _mcp_children()
        os.kill(child, signal.SIGKILL)
        results = [get() for _ in range(3)]
        assert not results[-1].is_error, results  # at most two transient failures
        assert all(r.error.retryable for r in results if r.error)
        assert _mcp_children() and child not in _mcp_children()
    finally:
        gateway.close()


class _SlowTeardownGateway:
    """Stalls the handshake, and takes a while to tear down when cancelled (a stdio child)."""

    lost = False
    opened = 0
    torn: list[int] = []

    async def __aenter__(self) -> _SlowTeardownGateway:
        type(self).opened += 1
        attempt = self.opened
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            await asyncio.sleep(0.5 if attempt == 1 else 0)  # the first outlasts the retry
            self.torn.append(attempt)
            raise
        raise AssertionError("unreachable")

    async def __aexit__(self, *exc: object) -> None:
        pass


def test_close_waits_for_abandoned_handshakes_to_tear_down() -> None:
    _SlowTeardownGateway.opened, _SlowTeardownGateway.torn = 0, []
    gateway = wiring.BackgroundMcpGateway(
        lambda: cast(McpToolGateway, _SlowTeardownGateway()),
        connect_timeout_s=0.05,
        close_timeout_s=2,
    )
    for _ in range(2):  # the retry starts a second owner while the first tears down
        result = asyncio.run(gateway.call(ToolName.GET_ORDER, {"order_id": "o"}, ctx()))
        assert result.error and result.error.category is ErrorCategory.TRANSIENT
    gateway.close()
    assert sorted(_SlowTeardownGateway.torn) == [1, 2]


def test_shutdown_settles_a_session_whose_owner_never_started() -> None:
    gateway = wiring.BackgroundMcpGateway(lambda: cast(McpToolGateway, _StalledGateway()))

    async def connect_then_shut_down() -> BaseException | None:
        session = gateway._connect()  # the owner task is created but hasn't run yet
        await gateway._shutdown()
        assert session.done()
        return session.exception()

    try:
        future = asyncio.run_coroutine_threadsafe(connect_then_shut_down(), gateway._loop)
        assert isinstance(future.result(timeout=5), ConnectionError)
    finally:
        gateway.close()


class _HangingCallGateway:
    """Connects, but a call never returns, even after the session ends."""

    lost = False

    async def __aenter__(self) -> _HangingCallGateway:
        return self

    async def __aexit__(self, *exc: object) -> None:
        pass

    async def wait_lost(self) -> None:
        await asyncio.sleep(3600)

    async def call(self, *args: object) -> ToolResult:
        await asyncio.sleep(3600)
        raise AssertionError("unreachable")


def test_a_call_cancelled_by_close_is_a_transient_result() -> None:
    gateway = wiring.BackgroundMcpGateway(
        lambda: cast(McpToolGateway, _HangingCallGateway()), close_timeout_s=0.2
    )

    async def call_then_close() -> ToolResult:
        pending = asyncio.ensure_future(gateway.call(ToolName.GET_ORDER, {"order_id": "o"}, ctx()))
        await asyncio.sleep(0.1)
        await asyncio.to_thread(gateway.close)
        return await pending

    result = asyncio.run(call_then_close())
    assert result.error and result.error.category is ErrorCategory.TRANSIENT
