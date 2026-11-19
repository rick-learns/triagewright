"""``ToolGateway`` implementations: in-process (direct) and an MCP client (stdio or HTTP).

Both log every raw call as a ``ToolCallRecord`` (including faults) and inject
eval ``ToolFault``s before the call leaves the gateway, like
``testing.FakeToolGateway``. Neither retries: lane B's tool-call wrapper does.
"""

from __future__ import annotations

import os
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
from triagewright.mcp_server.store import MockSystems


@dataclass
class _ArmedFault:
    fault: ToolFault
    remaining: int


class _RecordingGateway:
    """Fault injection and the raw call log. Subclasses implement ``_send``."""

    def __init__(self, faults: Sequence[ToolFault] = ()) -> None:
        self._faults = [_ArmedFault(f, f.times) for f in faults]
        self.calls: list[ToolCallRecord] = []

    def list_tools(self) -> Sequence[ToolSpec]:
        return list(TOOL_SPECS.values())

    async def call(self, name: ToolName, arguments: JsonObject, ctx: CallContext) -> ToolResult:
        started = time.perf_counter()
        result = self._fault_for(name, arguments)
        if result is None:
            result = await self._send(name, arguments, ctx)
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

    def __init__(self, service: ToolService, *, faults: Sequence[ToolFault] = ()) -> None:
        super().__init__(faults)
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
    because the server dedupes on the idempotency key.
    """

    def __init__(self, client: Client, *, faults: Sequence[ToolFault] = ()) -> None:
        super().__init__(faults)
        self._client = client

    @classmethod
    def from_settings(
        cls, settings: Settings, *, faults: Sequence[ToolFault] = ()
    ) -> McpToolGateway:
        """``http`` connects to ``settings.mcp_url``; ``stdio`` launches ``triagewright-mcp``."""
        if settings.mcp_transport == "http":
            return cls(Client(settings.mcp_url), faults=faults)
        if settings.mcp_transport == "stdio":
            return cls(Client(stdio_server_parameters()), faults=faults)
        raise ValueError("mcp_transport=inprocess uses InProcessToolGateway")

    async def __aenter__(self) -> Self:
        await self._client.__aenter__()
        try:
            listed = await self._client.list_tools()
            served = {t.name for t in listed.tools}
            if served != set(TOOL_SPECS):
                raise RuntimeError(f"MCP server tools {sorted(served)} != contract tools")
        except BaseException:
            await self._client.__aexit__(None, None, None)  # not exc_info: keep the error plain
            raise
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self._client.__aexit__(exc_type, exc, tb)

    async def _send(self, name: ToolName, arguments: JsonObject, ctx: CallContext) -> ToolResult:
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


def stdio_server_parameters() -> StdioServerParameters:
    """Launch this package's MCP server over stdio, passing the ``TW_*`` settings through."""
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "triagewright.mcp_server", "--transport", "stdio"],
        env={k: v for k, v in os.environ.items() if k.startswith("TW_")},
    )
