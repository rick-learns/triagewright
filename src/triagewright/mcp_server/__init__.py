"""Lane A: the MCP server over the mock business systems, and the gateways that reach it.

Wire contract
-------------

- Tools: the eight in ``contracts.tools.TOOL_SPECS``, with each input model's
  JSON Schema as ``inputSchema``.
- Call context: every ``tools/call`` carries the trusted
  :class:`~triagewright.contracts.tools.CallContext` in the request ``_meta``
  under the key ``"triagewright/ctx"``, as the JSON object
  ``CallContext.model_dump(mode="json")``. Code sets it from the ticket, never
  a model. A call without a valid one fails with ``permission``.
- Result: a ``CallToolResult`` whose ``isError`` equals
  ``ToolResult.is_error``, whose ``structuredContent`` is
  ``ToolResult.model_dump(mode="json")``, and whose single ``TextContent``
  block holds the same JSON (``common.canonical_json``). Tool failures are
  envelopes with a category and ``retryable``, never protocol errors. Only an
  unknown tool name is a JSON-RPC error (``INVALID_PARAMS``).

Run it with ``triagewright-mcp --transport stdio|http --host H --port P``. The
data lives in SQLite at ``TW_DB_PATH``, seeded from ``TW_SEED_DIR`` on first
start, with the clock pinned to the seed's ``as_of``. ``TW_APPROVAL_KEY`` (at
least 32 characters) is required: ``issue_refund`` verifies its HMAC token.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

import anyio
from pydantic import ValidationError

from triagewright.config import Settings
from triagewright.data import load_as_of, load_world
from triagewright.mcp_server.gateway import InProcessToolGateway, McpToolGateway
from triagewright.mcp_server.server import CTX_META_KEY, build_server, http_app, run_stdio
from triagewright.mcp_server.service import ToolService
from triagewright.mcp_server.store import MockSystems
from triagewright.rag import build_retriever

__all__ = [
    "CTX_META_KEY",
    "InProcessToolGateway",
    "McpToolGateway",
    "MockSystems",
    "ToolService",
    "build_server",
    "build_service",
    "main",
]


def build_service(settings: Settings, *, approval_key: bytes) -> ToolService:
    """The tool service over ``settings.db_path``, seeded from ``settings.seed_dir`` if new.

    KB search is lane C's hybrid retriever, indexed into the same database.
    """
    return ToolService(
        MockSystems.open(settings.db_path, seed=load_world(settings.seed_dir)),
        as_of=load_as_of(settings.seed_dir),
        approval_key=approval_key,
        retriever=build_retriever(settings),
        policy=settings.policy_config(),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="triagewright-mcp", description=__doc__.splitlines()[0])
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--no-env-file", action="store_true", help="read settings from the environment only"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        settings = Settings(_env_file=None) if args.no_env_file else Settings()
    except ValidationError as exc:  # report only the messages: inputs may hold secrets
        raise SystemExit("; ".join(e["msg"] for e in exc.errors())) from None
    if settings.approval_key is None:
        raise SystemExit("TW_APPROVAL_KEY (at least 32 characters) is required to serve MCP")
    logging.basicConfig(level=logging.INFO)  # stderr; stdout is the stdio transport
    service = build_service(
        settings, approval_key=settings.approval_key.get_secret_value().encode()
    )
    server = build_server(service)
    if args.transport == "stdio":
        anyio.run(run_stdio, server)
    else:
        import uvicorn

        uvicorn.run(http_app(server, host=args.host), host=args.host, port=args.port)
