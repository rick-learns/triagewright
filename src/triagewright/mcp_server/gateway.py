"""``ToolGateway`` implementations: in-process (direct) and an MCP client (stdio or HTTP).

Both log every raw call as a ``ToolCallRecord`` (including faults) and inject
eval ``ToolFault``s before the call leaves the gateway, like
``testing.FakeToolGateway``. Neither retries: lane B's tool-call wrapper does.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from types import TracebackType
from typing import Any, Self, cast

import anyio
import httpx
import mcp_types
from anyio.abc import TaskGroup, TaskStatus
from mcp import Client, MCPError, StdioServerParameters
from pydantic import ValidationError

from triagewright.config import Settings
from triagewright.contracts.common import JsonObject
from triagewright.contracts.evals import ToolFault
from triagewright.contracts.guardrails import PolicyConfig
from triagewright.contracts.ports import Retriever
from triagewright.contracts.tools import (
    TOOL_SPECS,
    CallContext,
    ErrorCategory,
    ToolCallRecord,
    ToolName,
    ToolResult,
    ToolSpec,
)
from triagewright.contracts.world import WorldState
from triagewright.mcp_server.server import CTX_META_KEY
from triagewright.mcp_server.service import ToolService
from triagewright.mcp_server.store import MEMORY, MockSystems

logger = logging.getLogger(__name__)


@dataclass
class _ArmedFault:
    fault: ToolFault
    remaining: int


class _RecordingGateway:
    """Fault injection and the raw call log. Subclasses implement ``_send``.

    ``log_calls=False`` keeps no log, for long-running processes (the API).
    """

    def __init__(self, faults: Sequence[ToolFault] = (), *, log_calls: bool = True) -> None:
        self._faults = [_ArmedFault(f, f.times) for f in faults]
        self._log_calls = log_calls
        self.calls: list[ToolCallRecord] = []

    def list_tools(self) -> Sequence[ToolSpec]:
        return list(TOOL_SPECS.values())

    async def call(self, name: ToolName, arguments: JsonObject, ctx: CallContext) -> ToolResult:
        started = time.perf_counter()
        result = self._fault_for(name, arguments)
        if result is None:
            result = await self._send(name, arguments, ctx)
        if not self._log_calls:
            return result
        self.calls.append(
            ToolCallRecord(
                tool=name,
                arguments=dict(arguments),
                caller=ctx.caller,
                result=result,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        )
        return result

    async def _send(
        self, name: ToolName, arguments: JsonObject, ctx: CallContext
    ) -> ToolResult:  # pragma: no cover - abstract
        raise NotImplementedError

    def _fault_for(self, name: ToolName, arguments: JsonObject) -> ToolResult | None:
        for armed in self._faults:
            fault = armed.fault
            if fault.tool is not name or armed.remaining <= 0:
                continue
            if fault.match_arguments and any(
                arguments.get(k) != v for k, v in fault.match_arguments.items()
            ):
                continue
            armed.remaining -= 1
            return ToolResult(tool=name, is_error=True, error=fault.error)
        return None


class InProcessToolGateway(_RecordingGateway):
    """Calls the tool service directly (``mcp_transport=inprocess``, tests, eval cases)."""

    def __init__(
        self, service: ToolService, *, faults: Sequence[ToolFault] = (), log_calls: bool = True
    ) -> None:
        super().__init__(faults, log_calls=log_calls)
        self.service = service

    @classmethod
    def for_world(
        cls,
        world: WorldState,
        *,
        as_of: datetime,
        approval_key: bytes,
        retriever: Retriever | None = None,
        faults: Sequence[ToolFault] = (),
        policy: PolicyConfig | None = None,
    ) -> InProcessToolGateway:
        """An isolated stack over a fresh in-memory database holding ``world``."""
        service = ToolService(
            MockSystems.in_memory(world),
            as_of=as_of,
            approval_key=approval_key,
            retriever=retriever,
            policy=policy,
        )
        return cls(service, faults=faults)

    def ledger(self) -> WorldState:
        return self.service.systems.ledger()

    async def _send(self, name: ToolName, arguments: JsonObject, ctx: CallContext) -> ToolResult:
        return await self.service.call(name, arguments, ctx)


_TRANSPORT_ERRORS = (
    OSError,
    httpx.TransportError,
    anyio.ClosedResourceError,
    anyio.BrokenResourceError,
    anyio.EndOfStream,
)


class McpToolGateway(_RecordingGateway):
    """A ``ToolGateway`` over an MCP client session. Use as ``async with``.

    ``CallContext`` travels in the request ``_meta`` under ``CTX_META_KEY``.
    Transport failures (connection lost, timeout) become retryable ``transient``
    or ``timeout`` results; a lost ``issue_refund`` response is safe to retry
    because the server dedupes on the idempotency key. The SDK session runs in
    its own task, so a dead transport fails calls instead of cancelling the
    caller. It doesn't reconnect: once the session is lost, every call is
    ``transient`` until the owner opens a new gateway.
    """

    def __init__(
        self, client: Client, *, faults: Sequence[ToolFault] = (), log_calls: bool = True
    ) -> None:
        super().__init__(faults, log_calls=log_calls)
        self._client = client
        self._stop = anyio.Event()
        self._lost = anyio.Event()
        self._tg: TaskGroup | None = None

    @classmethod
    def from_settings(
        cls, settings: Settings, *, faults: Sequence[ToolFault] = (), log_calls: bool = True
    ) -> McpToolGateway:
        """``http`` connects to ``settings.mcp_url``; ``stdio`` launches ``triagewright-mcp``."""
        if settings.mcp_transport == "http":
            return cls(Client(settings.mcp_url), faults=faults, log_calls=log_calls)
        if settings.mcp_transport == "stdio":
            client = Client(stdio_server_parameters(settings))
            return cls(client, faults=faults, log_calls=log_calls)
        raise ValueError("mcp_transport=inprocess uses InProcessToolGateway")

    @property
    def lost(self) -> bool:
        """The session ended. It doesn't reconnect: open a new gateway."""
        return self._lost.is_set()

    async def wait_lost(self) -> None:
        await self._lost.wait()

    async def __aenter__(self) -> Self:
        tg = anyio.create_task_group()
        await tg.__aenter__()
        try:
            await tg.start(self._run_session)
        except BaseException:
            await tg.__aexit__(None, None, None)
            raise
        self._tg = tg
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._stop.set()
        if self._tg is not None:
            await self._tg.__aexit__(None, None, None)

    async def _run_session(
        self, *, task_status: TaskStatus[None] = anyio.TASK_STATUS_IGNORED
    ) -> None:
        started = False
        try:
            async with self._client:
                # Prime the SDK's per-tool cache, so a call never triggers a tools/list
                # afterwards that could fail after the tool already ran.
                await self._client.list_tools()
                task_status.started()
                started = True
                await self._stop.wait()
        except Exception:
            if not started:
                raise
            logger.warning("MCP session lost", exc_info=True)
        finally:
            if started:
                self._lost.set()

    async def _send(self, name: ToolName, arguments: JsonObject, ctx: CallContext) -> ToolResult:
        done: list[ToolResult] = []

        async def call() -> None:
            done.append(await self._call_tool(name, arguments, ctx))
            tg.cancel_scope.cancel()

        async def watch() -> None:
            await self._lost.wait()
            tg.cancel_scope.cancel()

        async with anyio.create_task_group() as tg:
            tg.start_soon(call)
            tg.start_soon(watch)
        if done:
            return done[0]
        return ToolResult.fail(name, ErrorCategory.TRANSIENT, "MCP connection lost")

    async def _call_tool(
        self, name: ToolName, arguments: JsonObject, ctx: CallContext
    ) -> ToolResult:
        meta = cast(mcp_types.RequestParamsMeta, {CTX_META_KEY: ctx.model_dump(mode="json")})
        try:
            raw = await self._client.call_tool(name.value, dict(arguments), meta=meta)
        except MCPError as exc:
            if exc.code == mcp_types.REQUEST_TIMEOUT:
                return ToolResult.fail(name, ErrorCategory.TIMEOUT, "MCP request timed out")
            if exc.code == mcp_types.CONNECTION_CLOSED:
                return ToolResult.fail(name, ErrorCategory.TRANSIENT, "MCP connection closed")
            return ToolResult.fail(name, ErrorCategory.INTERNAL, f"MCP error {exc.code}")
        except TimeoutError:
            return ToolResult.fail(name, ErrorCategory.TIMEOUT, "MCP request timed out")
        except _TRANSPORT_ERRORS as exc:
            return ToolResult.fail(
                name, ErrorCategory.TRANSIENT, f"MCP transport error: {type(exc).__name__}"
            )
        return _parse_result(name, raw)


def _parse_result(name: ToolName, raw: mcp_types.CallToolResult) -> ToolResult:
    payload: Any = raw.structured_content
    try:
        if payload is None:
            text = next(b.text for b in raw.content if isinstance(b, mcp_types.TextContent))
            result = ToolResult.model_validate_json(text)
        else:
            result = ToolResult.model_validate(payload)
    except (StopIteration, ValidationError):
        return ToolResult.fail(name, ErrorCategory.INTERNAL, "malformed MCP tool result")
    if result.tool is not name or result.is_error != raw.is_error:
        return ToolResult.fail(name, ErrorCategory.INTERNAL, "inconsistent MCP tool result")
    return result


def stdio_server_parameters(settings: Settings) -> StdioServerParameters:
    """Launch this package's MCP server over stdio with the server-side ``settings``."""
    if settings.approval_key is None:
        raise ValueError("TW_APPROVAL_KEY is required to launch the MCP server")
    db = str(settings.db_path) if str(settings.db_path) == MEMORY else settings.db_path.resolve()
    # The child gets exactly the parent's resolved server settings: no ambient TW_* env (the SDK
    # passes only a safe default env) and no .env, so it can't reject what the parent accepted.
    env = {
        "TW_APPROVAL_KEY": settings.approval_key.get_secret_value(),
        "TW_DB_PATH": str(db),
        "TW_SEED_DIR": str(settings.seed_dir.resolve()),
        "TW_KB_DIR": str(settings.kb_dir.resolve()),
        "TW_REFUND_HARD_LIMIT_CENTS": str(settings.refund_hard_limit_cents),
        "TW_REFUND_WINDOW_DAYS": str(settings.refund_window_days),
        "TW_MAX_REFUNDS_PER_CUSTOMER_30D": str(settings.max_refunds_per_customer_30d),
    }
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "triagewright.mcp_server", "--transport", "stdio", "--no-env-file"],
        env=env,
    )
