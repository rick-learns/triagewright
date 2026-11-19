"""The composition root, end to end on the real lanes with a scripted model (no keys)."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest
from mcp import Client

from tests.agent.helpers import call, finalize, refund_script, review, triage
from tests.mcp_server.conftest import ctx
from triagewright import wiring
from triagewright.api.services import (
    DecisionConflictError,
    Services,
    WiringUnavailableError,
    decide_approval,
)
from triagewright.config import Settings
from triagewright.contracts.agents import ResolutionStatus
from triagewright.contracts.llm import ModelRequest, ModelResponse
from triagewright.contracts.ports import CaseRuntime, CaseRuntimeFactory
from triagewright.contracts.tools import ErrorCategory, ToolName
from triagewright.evals import cli as evals_cli
from triagewright.llm import ModelConfigError, ReplayClient, build_model_clients
from triagewright.mcp_server import McpToolGateway, MockSystems, ToolService, build_server
from triagewright.testing import (
    TEST_APPROVAL_KEY,
    FakeRetriever,
    ScriptedModelClient,
    sample_ticket,
    seed_as_of,
    seed_kb,
    seed_world,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


REFUND_TICKET = sample_ticket("Please refund $50.00 for order ord_a1001, it arrived broken.")


def settings(tmp_path: Path, **overrides: object) -> Settings:
    return Settings(
        _env_file=None,
        db_path=tmp_path / "world.sqlite",
        approvals_db_path=tmp_path / "approvals.sqlite",
        trace_exporter="none",
        **overrides,  # type: ignore[arg-type]
    )


class Scripted:
    """Scripted responses, stamped with the requested model so the real price table applies."""

    def __init__(self, responses: Sequence[ModelResponse]) -> None:
        self.inner = ScriptedModelClient(responses)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        response = await self.inner.complete(request)
        return response.model_copy(update={"model": request.model})


def scripted(responses: Sequence[ModelResponse]) -> wiring.ModelFactory:
    return lambda case_id: Scripted(responses)


# ---- evals -------------------------------------------------------------------------


async def test_eval_stack_runs_a_refund_through_the_real_executor(tmp_path: Path) -> None:
    factory = wiring.EvalStackFactory(
        settings(tmp_path), model_factory=scripted(refund_script("ord_a1001", 5000))
    )
    assert isinstance(factory, CaseRuntimeFactory)
    case = factory.build(case_id="policy-001", world=seed_world(), as_of=seed_as_of(), faults=())
    assert isinstance(case, CaseRuntime)

    response = await case.resolver.resolve(REFUND_TICKET)

    assert response.status is ResolutionStatus.PENDING_APPROVAL, response
    assert response.metrics is not None and response.metrics.estimated  # replay mode
    (pending,) = case.pending_approvals()
    assert pending.tool is ToolName.ISSUE_REFUND and pending.arguments["amount_cents"] == 5000
    assert case.ledger().refunds == seed_world().refunds  # approval never completes in-request
    lookups = [c for c in case.tool_calls() if c.tool is ToolName.CHECK_REFUND_ELIGIBILITY]
    assert [c.caller for c in lookups] == ["action"]
    assert not (tmp_path / "world.sqlite").exists()  # evals stay in memory


async def test_eval_cases_are_isolated_from_each_other(tmp_path: Path) -> None:
    factory = wiring.EvalStackFactory(
        settings(tmp_path), model_factory=scripted(refund_script("ord_a1001", 5000))
    )
    first = factory.build(case_id="policy-001", world=seed_world(), as_of=seed_as_of(), faults=())
    await first.resolver.resolve(REFUND_TICKET)
    second = factory.build(case_id="policy-002", world=seed_world(), as_of=seed_as_of(), faults=())
    assert first.pending_approvals() and not second.pending_approvals()
    assert not second.tool_calls()


async def test_eval_stack_applies_case_faults(tmp_path: Path) -> None:
    from triagewright.contracts.evals import ToolFault
    from triagewright.contracts.tools import ToolError

    fault = ToolFault(
        tool=ToolName.GET_ORDER,
        error=ToolError(category=ErrorCategory.TRANSIENT, message="down", retryable=True),
    )
    script = [
        triage(),
        call("get_order", {"order_id": "ord_a1002"}),
        finalize(),
        review(),
    ]
    factory = wiring.EvalStackFactory(settings(tmp_path), model_factory=scripted(script))
    case = factory.build(
        case_id="toolfail-001", world=seed_world(), as_of=seed_as_of(), faults=[fault]
    )
    await case.resolver.resolve(sample_ticket())
    attempts = [c for c in case.tool_calls() if c.tool is ToolName.GET_ORDER]
    assert [c.result.is_error for c in attempts] == [True, False]  # faulted once, then retried


def test_case_runtime_factory_reads_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(wiring, "get_settings", lambda: settings(tmp_path))
    assert isinstance(wiring.case_runtime_factory(), wiring.EvalStackFactory)


def test_evals_cli_runs_the_default_factory_in_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No recordings: the model call misses, the agent hands off, and the run still reports."""
    empty = settings(tmp_path, recordings_dir=tmp_path / "recordings")
    monkeypatch.setattr(wiring, "get_settings", lambda: empty)
    monkeypatch.setattr(evals_cli, "get_settings", lambda: empty)
    code = evals_cli.main(
        [
            "run",
            "--case",
            "happy-001",
            "--report",
            str(tmp_path / "report.md"),
            "--json",
            str(tmp_path / "report.json"),
            "--results",
            str(tmp_path / "results.jsonl"),
            "--baseline",
            str(tmp_path / "baseline.json"),
        ]
    )
    assert code in (0, 1)
    assert "1 cases (replay)" in capsys.readouterr().out
    assert (tmp_path / "report.md").exists()


def test_judge_client_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    current = settings(tmp_path, recordings_dir=tmp_path)
    monkeypatch.setattr(wiring, "get_settings", lambda: current)
    build = wiring.judge_client_factory()
    client = build(case_id="happy-001", recordings=tmp_path / "happy-001.judge.jsonl")
    assert isinstance(client, ReplayClient) and client.model == current.models()["judge"]
    with pytest.raises(ModelConfigError, match="judge recordings"):
        build(case_id="happy-001", recordings=tmp_path / "happy-001.jsonl")


# ---- API and CLI -------------------------------------------------------------------


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


def test_role_router_rejects_an_unknown_purpose(tmp_path: Path) -> None:
    clients = build_model_clients(settings(tmp_path), case_id="happy-001")
    router = wiring.RoleRouter(clients)
    request = ModelRequest.model_validate(
        {
            "model": "gemini-3.5-flash-lite",
            "system": "s",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "purpose": "nope",
        }
    )
    with pytest.raises(ModelConfigError, match="purpose 'nope'"):
        asyncio.run(router.complete(request))


# ---- MCP over a background loop ----------------------------------------------------


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


def test_services_reach_an_out_of_process_mcp_server_lazily(tmp_path: Path) -> None:
    remote = settings(tmp_path, mcp_transport="http", approval_key="k" * 32)
    built = wiring.build_services(remote, model_factory=scripted(()))
    gateway = built.executor.gateway  # type: ignore[attr-defined]
    assert isinstance(gateway, wiring.BackgroundMcpGateway)  # no connection until a call
    assert {s.name for s in gateway.list_tools()} == set(ToolName)
    gateway.close()
    assert not (tmp_path / "world.sqlite").exists()  # the MCP server owns the world


def test_live_clients_share_one_http_pool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    live = settings(
        tmp_path,
        llm_provider="gemini",
        gemini_api_key="test-key-not-real",
        recordings_dir=tmp_path / "recordings",
    )
    monkeypatch.setattr(wiring, "get_settings", lambda: live)
    build = wiring.judge_client_factory()
    one = build(case_id="happy-001", recordings=tmp_path / "recordings" / "happy-001.judge.jsonl")
    two = build(case_id="happy-002", recordings=tmp_path / "recordings" / "happy-002.judge.jsonl")
    assert one.model == two.model == live.models()["judge"]  # type: ignore[attr-defined]
    assert one._http is two._http  # type: ignore[attr-defined]
