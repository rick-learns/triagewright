"""The MCP server: the contract tools over a :class:`ToolService`, on the low-level ``mcp`` SDK.

See the package docstring for the wire contract.
"""

from __future__ import annotations

from typing import Any, Final

import mcp_types as types
from mcp import MCPError
from mcp.server import Server
from mcp.server.context import ServerRequestContext
from mcp.server.stdio import stdio_server
from pydantic import ValidationError
from starlette.applications import Starlette

from triagewright.contracts.common import JsonObject, canonical_json
from triagewright.contracts.tools import (
    TOOL_SPECS,
    CallContext,
    ErrorCategory,
    ToolName,
    ToolResult,
)
from triagewright.mcp_server.service import ToolService

CTX_META_KEY: Final = "triagewright/ctx"
SERVER_NAME: Final = "triagewright"


def _tools() -> list[types.Tool]:
    return [
        types.Tool(
            name=spec.name.value, description=spec.description, input_schema=spec.input_schema()
        )
        for spec in TOOL_SPECS.values()
    ]


def to_call_tool_result(result: ToolResult) -> types.CallToolResult:
    """The envelope as ``structuredContent`` plus the same JSON in one text block."""
    payload = result.model_dump(mode="json")
    return types.CallToolResult(
        content=[types.TextContent(text=canonical_json(payload))],
        structured_content=payload,
        is_error=result.is_error,
    )


def _call_context(meta: types.RequestParamsMeta | None) -> CallContext | None:
    raw = None if meta is None else meta.get(CTX_META_KEY)
    try:
        return CallContext.model_validate(raw)
    except ValidationError:
        return None


def build_server(service: ToolService) -> Server[Any]:
    tools = _tools()

    async def list_tools(
        ctx: ServerRequestContext[Any], params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        return types.ListToolsResult(tools=tools)

    async def call_tool(
        ctx: ServerRequestContext[Any], params: types.CallToolRequestParams
    ) -> types.CallToolResult:
        try:
            name = ToolName(params.name)
        except ValueError:
            raise MCPError(types.INVALID_PARAMS, f"unknown tool {params.name!r}") from None
        call_ctx = _call_context(params.meta)
        if call_ctx is None:
            result = ToolResult.fail(
                name, ErrorCategory.PERMISSION, f"missing or invalid {CTX_META_KEY} in _meta"
            )
        else:
            arguments: JsonObject = dict(params.arguments or {})
            result = await service.call(name, arguments, call_ctx)
        return to_call_tool_result(result)

    return Server(SERVER_NAME, on_list_tools=list_tools, on_call_tool=call_tool)


async def run_stdio(server: Server[Any]) -> None:
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def http_app(server: Server[Any], *, host: str) -> Starlette:
    """Streamable HTTP at ``/mcp``. Stateless JSON responses: each tool call stands alone."""
    return server.streamable_http_app(host=host, stateless_http=True, json_response=True)
