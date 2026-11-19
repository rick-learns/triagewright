from __future__ import annotations

import sys
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest
from mcp import Client, StdioServerParameters

from triagewright.contracts.evals import ToolFault
from triagewright.contracts.ports import ToolGateway
from triagewright.contracts.tools import CallContext, Caller, ToolCallRecord
from triagewright.contracts.world import WorldState
from triagewright.mcp_server import InProcessToolGateway, McpToolGateway, MockSystems, ToolService
from triagewright.mcp_server.server import build_server
from triagewright.testing import (
    REPO_ROOT,
    TEST_APPROVAL_KEY,
    FakeRetriever,
    FakeToolGateway,
    seed_as_of,
    seed_kb,
    seed_world,
)

# Out-of-process servers need TW_APPROVAL_KEY of at least 32 characters.
SUBPROCESS_KEY = "triagewright-integration-test-key-0123456789"

KINDS = [
    "fake",
    "inprocess",
    "mcp-memory",
    pytest.param("mcp-stdio", marks=pytest.mark.integration),
]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def ctx(customer_id: str | None = "cus_ava01", caller: Caller = "action") -> CallContext:
    return CallContext(
        request_id="req_00000001", ticket_id="tkt_0001", customer_id=customer_id, caller=caller
    )


@dataclass
class Harness:
    """One gateway over a fresh seed world, plus what a grader can inspect."""

    gateway: ToolGateway
    key: bytes
    calls: Callable[[], list[ToolCallRecord]]
    ledger: Callable[[], WorldState]


def subprocess_env(tmp_path: Path) -> dict[str, str]:
    return {
        "TW_DB_PATH": str(tmp_path / "mock.sqlite"),
        "TW_APPROVAL_KEY": SUBPROCESS_KEY,
        "TW_SEED_DIR": str(REPO_ROOT / "data" / "seed"),
        "TW_KB_DIR": str(REPO_ROOT / "data" / "kb"),
    }


def new_service() -> ToolService:
    return ToolService(
        MockSystems.in_memory(seed_world()),
        as_of=seed_as_of(),
        approval_key=TEST_APPROVAL_KEY,
        retriever=FakeRetriever(seed_kb()),
    )


@asynccontextmanager
async def open_harness(
    kind: str, tmp_path: Path, faults: Sequence[ToolFault] = ()
) -> AsyncIterator[Harness]:
    if kind == "fake":
        fake = FakeToolGateway(
            seed_world(), as_of=seed_as_of(), retriever=FakeRetriever(seed_kb()), faults=faults
        )
        yield Harness(fake, TEST_APPROVAL_KEY, lambda: fake.calls, fake.ledger)
    elif kind == "inprocess":
        direct = InProcessToolGateway.for_world(
            seed_world(),
            as_of=seed_as_of(),
            approval_key=TEST_APPROVAL_KEY,
            retriever=FakeRetriever(seed_kb()),
            faults=faults,
        )
        yield Harness(direct, TEST_APPROVAL_KEY, lambda: direct.calls, direct.ledger)
    elif kind == "mcp-memory":
        service = new_service()
        # "legacy" runs the JSON-RPC stream loop; the default would dispatch directly.
        client = Client(build_server(service), mode="legacy")
        async with McpToolGateway(client, faults=faults) as gw:
            yield Harness(gw, TEST_APPROVAL_KEY, lambda: gw.calls, service.systems.ledger)
    elif kind == "mcp-stdio":
        env = subprocess_env(tmp_path)
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "triagewright.mcp_server", "--transport", "stdio"],
            env=env,
            cwd=str(REPO_ROOT),
        )

        def ledger() -> WorldState:
            systems = MockSystems.open(env["TW_DB_PATH"], seed=WorldState())
            try:
                return systems.ledger()
            finally:
                systems.close()

        async with McpToolGateway(Client(params), faults=faults) as gw:
            yield Harness(gw, SUBPROCESS_KEY.encode(), lambda: gw.calls, ledger)
    else:  # pragma: no cover
        raise ValueError(kind)


@pytest.fixture(params=KINDS)
def kind(request: pytest.FixtureRequest) -> str:
    value: str = request.param
    return value
