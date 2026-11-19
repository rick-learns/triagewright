"""The API/CLI services: building, persisted approvals, and closing what they own."""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.agent.helpers import refund_script
from tests.mcp_server.conftest import ctx
from tests.wiring.helpers import REFUND_TICKET, memory_gateway, scripted, settings
from triagewright import wiring
from triagewright.api.services import (
    DecisionConflictError,
    Services,
    WiringUnavailableError,
    decide_approval,
)
from triagewright.config import Settings
from triagewright.contracts.agents import ResolutionStatus
from triagewright.contracts.llm import ModelResponse
from triagewright.contracts.rag import RetrievalQuery
from triagewright.contracts.tools import ToolName
from triagewright.mcp_server import McpToolGateway, MockSystems
from triagewright.testing import seed_world

pytestmark = pytest.mark.anyio


def test_services_refuse_replay_without_a_model(tmp_path: Path) -> None:
    with pytest.raises(WiringUnavailableError, match="replay only replays"):
        wiring.build_services(settings(tmp_path))


def services(tmp_path: Path, responses: Sequence[ModelResponse] = ()) -> Services:
    return wiring.build_services(settings(tmp_path), model_factory=scripted(responses))


async def test_services_persist_approvals_and_execute_on_a_human_decision(
    tmp_path: Path,
) -> None:
    first = services(tmp_path, refund_script("ord_a1001", 5000))
    response = await first.resolver.resolve(REFUND_TICKET)
    assert response.status is ResolutionStatus.PENDING_APPROVAL, response
    (pending,) = first.approvals.list_pending()

    # A restart: new process, new ephemeral approval key, same databases.
    second = services(tmp_path)
    assert [a.approval_id for a in second.approvals.list_pending()] == [pending.approval_id]
    now = datetime(2026, 6, 15, 13, tzinfo=UTC)
    outcome = await decide_approval(
        second, pending.approval_id, approved=True, reviewer="api", note="ok", now=now
    )
    assert outcome.status.value == "executed", outcome
    refunds = MockSystems.open(tmp_path / "world.sqlite", seed=seed_world()).refunds_of("cus_ava01")
    assert [r.amount_cents for r in refunds if r.order_id == "ord_a1001"][-1] == 5000
    with pytest.raises(DecisionConflictError):
        await decide_approval(
            second, pending.approval_id, approved=False, reviewer="api", note="", now=now
        )


def test_services_reach_an_out_of_process_mcp_server_lazily(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = settings(tmp_path, mcp_transport="http", approval_key="k" * 32)
    opened: list[Settings] = []

    def from_settings(current: Settings, *, log_calls: bool) -> McpToolGateway:
        opened.append(current)
        return memory_gateway()  # stands in for the server at mcp_url

    monkeypatch.setattr(McpToolGateway, "from_settings", from_settings)
    built = wiring.build_services(remote, model_factory=scripted(()))
    gateway = built.executor.gateway  # type: ignore[attr-defined]
    assert isinstance(gateway, wiring.BackgroundMcpGateway)
    assert {s.name for s in gateway.list_tools()} == set(ToolName)
    try:

        async def session() -> object:  # runs after anything building queued on the loop
            return gateway._session

        assert asyncio.run_coroutine_threadsafe(session(), gateway._loop).result(30) is None
        assert opened == []  # building doesn't connect
        result = asyncio.run(gateway.call(ToolName.GET_ORDER, {"order_id": "ord_a1002"}, ctx()))
        assert not result.is_error, result
        assert opened == [remote]  # the first call does
    finally:
        built.close()
    assert not (tmp_path / "world.sqlite").exists()  # the MCP server owns the world


def test_services_close_their_gateway_and_store(tmp_path: Path) -> None:
    remote = settings(tmp_path, mcp_transport="http", approval_key="k" * 32)
    built = wiring.build_services(remote, model_factory=scripted(()))
    built.close()
    gateway = built.executor.gateway  # type: ignore[attr-defined]
    closed = asyncio.run(gateway.call(ToolName.GET_ORDER, {"order_id": "o"}, ctx()))
    assert closed.error and closed.error.message == "MCP gateway closed"
    with pytest.raises(sqlite3.ProgrammingError):
        built.approvals.list_pending()


def test_services_close_release_the_world_and_kb_stores(tmp_path: Path) -> None:
    built = wiring.build_services(settings(tmp_path), model_factory=scripted(()))
    service = built.executor.gateway.service  # type: ignore[attr-defined]

    async def teardown() -> None:
        await built.aclose()

    asyncio.run(teardown())
    built.close()
    with pytest.raises(sqlite3.ProgrammingError):
        service.systems.transaction().__enter__()
    with pytest.raises(sqlite3.ProgrammingError):
        service.retriever.search(RetrievalQuery(text="refund"))
