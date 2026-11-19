from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, cast

import anyio
import mcp_types
import pytest
from mcp import Client, MCPError, StdioServerParameters

from tests.mcp_server.conftest import ctx, new_service
from triagewright.config import Settings
from triagewright.contracts.common import canonical_json
from triagewright.contracts.rag import RetrievalQuery, RetrievedChunk
from triagewright.contracts.tools import TOOL_SPECS, ErrorCategory, ToolName, ToolResult
from triagewright.mcp_server import CTX_META_KEY, McpToolGateway, ToolService
from triagewright.mcp_server.gateway import _parse_result, stdio_server_parameters
from triagewright.mcp_server.server import build_server, to_call_tool_result
from triagewright.mcp_server.store import MockSystems
from triagewright.testing import TEST_APPROVAL_KEY, seed_as_of, seed_world

pytestmark = pytest.mark.anyio


def meta(value: object) -> mcp_types.RequestParamsMeta:
    return cast(mcp_types.RequestParamsMeta, {CTX_META_KEY: value})


async def test_lists_tools_with_input_schemas() -> None:
    async with Client(build_server(new_service()), mode="legacy") as client:
        listed = await client.list_tools()
    assert [t.name for t in listed.tools] == [n.value for n in TOOL_SPECS]
    for tool in listed.tools:
        spec = TOOL_SPECS[ToolName(tool.name)]
        assert tool.input_schema == spec.input_schema() and tool.description == spec.description


async def test_result_is_the_envelope_with_is_error() -> None:
    wire = ctx().model_dump(mode="json")
    async with Client(build_server(new_service()), mode="legacy") as client:
        ok = await client.call_tool("get_order", {"order_id": "ord_a1001"}, meta=meta(wire))
        missing = await client.call_tool("get_order", {"order_id": "ord_zzzz"}, meta=meta(wire))
    assert not ok.is_error and missing.is_error
    for raw in (ok, missing):
        envelope = ToolResult.model_validate(raw.structured_content)
        assert envelope.is_error is raw.is_error
        assert len(raw.content) == 1 and isinstance(raw.content[0], mcp_types.TextContent)
        assert raw.content[0].text == canonical_json(raw.structured_content)
    err = json.loads(cast(mcp_types.TextContent, missing.content[0]).text)["error"]
    assert err == {
        "category": "not_found",
        "message": "ord_zzzz",
        "retryable": False,
        "retry_after_ms": None,
        "details": {},
    }


@pytest.mark.parametrize(
    "ctx_meta",
    [None, {"caller": "human"}, "not an object", {**ctx().model_dump(mode="json"), "x": 1}],
)
async def test_call_without_a_valid_context_is_refused(ctx_meta: object) -> None:
    async with Client(build_server(new_service()), mode="legacy") as client:
        raw = await client.call_tool(
            "get_order",
            {"order_id": "ord_a1001"},
            meta=None if ctx_meta is None else meta(ctx_meta),
        )
    envelope = ToolResult.model_validate(raw.structured_content)
    assert raw.is_error and envelope.error and envelope.error.category is ErrorCategory.PERMISSION


async def test_unknown_tool_is_a_protocol_error() -> None:
    async with Client(build_server(new_service()), mode="legacy") as client:
        with pytest.raises(MCPError) as info:
            await client.call_tool("drop_tables", {}, meta=meta(ctx().model_dump(mode="json")))
    assert info.value.code == mcp_types.INVALID_PARAMS


class _BrokenRetriever:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]:
        raise self.exc


@pytest.mark.parametrize(
    ("exc", "expected", "retryable"),
    [
        (sqlite3.OperationalError("database is locked"), ErrorCategory.TRANSIENT, True),
        (RuntimeError("bug"), ErrorCategory.INTERNAL, False),
    ],
)
async def test_handler_failures_become_envelopes(
    exc: Exception, expected: ErrorCategory, retryable: bool
) -> None:
    service = ToolService(
        MockSystems.in_memory(seed_world()),
        as_of=seed_as_of(),
        approval_key=TEST_APPROVAL_KEY,
        retriever=_BrokenRetriever(exc),
    )
    result = await service.call(ToolName.SEARCH_KB, {"query": "refund"}, ctx())
    assert result.error and result.error.category is expected
    assert result.error.retryable is retryable


async def test_search_without_a_retriever_returns_no_hits() -> None:
    service = ToolService(
        MockSystems.in_memory(seed_world()), as_of=seed_as_of(), approval_key=TEST_APPROVAL_KEY
    )
    result = await service.call(ToolName.SEARCH_KB, {"query": "refund"}, ctx(None))
    assert result.data == {"hits": []}


class _RaisingClient:
    def __init__(self, exc: BaseException) -> None:
        self.exc = exc

    async def call_tool(self, *args: Any, **kwargs: Any) -> mcp_types.CallToolResult:
        raise self.exc


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (MCPError(mcp_types.REQUEST_TIMEOUT, "slow"), ErrorCategory.TIMEOUT),
        (MCPError(mcp_types.CONNECTION_CLOSED, "gone"), ErrorCategory.TRANSIENT),
        (MCPError(mcp_types.INTERNAL_ERROR, "boom"), ErrorCategory.INTERNAL),
        (TimeoutError(), ErrorCategory.TIMEOUT),
        (ConnectionResetError(), ErrorCategory.TRANSIENT),
    ],
)
async def test_transport_failures_map_to_categories(
    exc: BaseException, expected: ErrorCategory
) -> None:
    gw = McpToolGateway(cast(Client, _RaisingClient(exc)))
    result = await gw.call(ToolName.GET_ORDER, {"order_id": "ord_a1001"}, ctx())
    assert result.error and result.error.category is expected
    assert result.error.retryable is (expected is not ErrorCategory.INTERNAL)
    assert len(gw.calls) == 1  # logged, and never retried by the gateway


@pytest.mark.parametrize(
    "exc",
    [MCPError(mcp_types.CONNECTION_CLOSED, "gone"), anyio.ClosedResourceError()],
)
async def test_a_terminal_disconnect_marks_the_session_lost(exc: BaseException) -> None:
    raising = _RaisingClient(exc)
    gw = McpToolGateway(cast(Client, raising))
    assert not gw.lost
    first = await gw.call(ToolName.GET_ORDER, {"order_id": "ord_a1001"}, ctx())
    assert first.error and first.error.category is ErrorCategory.TRANSIENT and gw.lost
    raising.exc = AssertionError("the SDK must not be called on a lost session")
    again = await gw.call(ToolName.GET_ORDER, {"order_id": "ord_a1001"}, ctx())
    assert again.error and again.error.category is ErrorCategory.TRANSIENT
    await gw.wait_lost()  # already set: returns at once


async def test_a_one_off_transport_error_keeps_the_session() -> None:
    gw = McpToolGateway(cast(Client, _RaisingClient(ConnectionResetError())))
    await gw.call(ToolName.GET_ORDER, {"order_id": "ord_a1001"}, ctx())
    assert not gw.lost  # an HTTP request can fail without ending the session


async def test_an_sdk_error_after_the_session_ends_is_a_transient_result() -> None:
    class ClosingClient:
        async def call_tool(self, *args: Any, **kwargs: Any) -> mcp_types.CallToolResult:
            gw._stop.set()  # the owner closes the gateway while this call is in the SDK
            raise RuntimeError("session closed")

    gw = McpToolGateway(cast(Client, ClosingClient()))
    result = await gw.call(ToolName.GET_ORDER, {"order_id": "ord_a1001"}, ctx())
    assert result.error and result.error.category is ErrorCategory.TRANSIENT


def test_parse_result_rejects_malformed_or_inconsistent_results() -> None:
    good = ToolResult.fail(ToolName.GET_ORDER, ErrorCategory.NOT_FOUND, "ord_zzzz")
    wire = to_call_tool_result(good)
    assert _parse_result(ToolName.GET_ORDER, wire) == good
    text_only = mcp_types.CallToolResult(content=wire.content, is_error=True)
    assert _parse_result(ToolName.GET_ORDER, text_only) == good
    bad = [
        (ToolName.GET_ORDER, mcp_types.CallToolResult(content=[], is_error=True)),
        (ToolName.GET_ORDER, mcp_types.CallToolResult(content=[], structured_content={"a": 1})),
        (ToolName.GET_ACCOUNT, wire),
        (ToolName.GET_ORDER, wire.model_copy(update={"is_error": False})),
    ]
    for name, raw in bad:
        parsed = _parse_result(name, raw)
        assert parsed.error and parsed.error.category is ErrorCategory.INTERNAL, raw


def test_from_settings_picks_the_transport(tmp_path: Path) -> None:
    http = Settings(
        _env_file=None, mcp_transport="http", mcp_url="http://mcp:8765/mcp", approval_key="k" * 32
    )
    assert McpToolGateway.from_settings(http)._client.server == "http://mcp:8765/mcp"
    stdio = Settings(
        _env_file=None,
        mcp_transport="stdio",
        approval_key="s" * 40,
        db_path=tmp_path / "db.sqlite",
        refund_window_days=14,
    )
    params = McpToolGateway.from_settings(stdio)._client.server
    assert isinstance(params, StdioServerParameters) and params.env is not None
    assert params.args[-3:] == ["--transport", "stdio", "--no-env-file"]
    # The child gets the resolved settings, not just whatever is in os.environ.
    assert params.env["TW_APPROVAL_KEY"] == "s" * 40
    assert params.env["TW_DB_PATH"] == str(tmp_path / "db.sqlite")
    assert params.env["TW_REFUND_WINDOW_DAYS"] == "14"
    assert Path(params.env["TW_SEED_DIR"]).is_absolute()
    unkeyed = Settings(_env_file=None, mcp_transport="inprocess", approval_key=None)
    with pytest.raises(ValueError, match="inprocess"):
        McpToolGateway.from_settings(unkeyed)
    with pytest.raises(ValueError, match="TW_APPROVAL_KEY"):
        stdio_server_parameters(unkeyed)


def test_stdio_child_gets_only_resolved_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    # The parent accepted these settings; ambient env the child could reject must not reach it.
    monkeypatch.setenv("TW_LLM_PROVIDER", "gemini")
    monkeypatch.setenv("TW_MODEL_TRIAGE", "example/example:free")
    settings = Settings(
        _env_file=None,
        llm_provider="replay",
        approval_key="s" * 40,
        db_path=Path(":memory:"),
        embedder="fastembed",
    )
    params = stdio_server_parameters(settings)
    assert params.env and params.env["TW_DB_PATH"] == ":memory:"
    assert params.env["TW_EMBEDDER"] == "fastembed"  # the child builds the same KB index
    assert not {"TW_LLM_PROVIDER", "TW_MODEL_TRIAGE"} & params.env.keys()
    assert params.args[-1] == "--no-env-file"


async def test_a_rejected_reentry_keeps_the_live_session() -> None:
    gw = McpToolGateway(Client(build_server(new_service()), mode="legacy"))
    async with gw:
        with pytest.raises(RuntimeError, match="already entered"):
            await gw.__aenter__()
        ok = await gw.call(ToolName.GET_ORDER, {"order_id": "ord_a1001"}, ctx())
        assert not ok.is_error
