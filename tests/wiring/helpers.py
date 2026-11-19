"""Settings and scripted models shared by the wiring tests."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from mcp import Client

from triagewright import wiring
from triagewright.config import Settings
from triagewright.contracts.llm import ModelRequest, ModelResponse
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

REFUND_TICKET = sample_ticket("Please refund $50.00 for order ord_a1001, it arrived broken.")


def settings(tmp_path: Path, **overrides: object) -> Settings:
    return Settings(
        _env_file=None,
        db_path=tmp_path / "world.sqlite",
        approvals_db_path=tmp_path / "approvals.sqlite",
        trace_exporter="none",
        **overrides,  # type: ignore[arg-type]
    )


def memory_gateway() -> McpToolGateway:
    """A real MCP client session over an in-memory server and world."""
    service = ToolService(
        MockSystems.in_memory(seed_world()),
        as_of=seed_as_of(),
        approval_key=TEST_APPROVAL_KEY,
        retriever=FakeRetriever(seed_kb()),
    )
    return McpToolGateway(Client(build_server(service), mode="legacy"))


class Scripted:
    """Scripted responses, stamped with the requested model so the real price table applies."""

    def __init__(self, responses: Sequence[ModelResponse]) -> None:
        self.inner = ScriptedModelClient(responses)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        response = await self.inner.complete(request)
        return response.model_copy(update={"model": request.model})


def scripted(responses: Sequence[ModelResponse]) -> wiring.ModelFactory:
    return lambda case_id: Scripted(responses)
