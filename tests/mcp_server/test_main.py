from __future__ import annotations

import socket
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import anyio
import pytest
from mcp import Client

from tests.mcp_server.conftest import SUBPROCESS_KEY, ctx, subprocess_env
from triagewright.contracts.tools import ErrorCategory, ToolName
from triagewright.mcp_server import McpToolGateway, main
from triagewright.mcp_server.server import run_stdio
from triagewright.testing import REPO_ROOT


@pytest.fixture
def server_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, str]:
    monkeypatch.chdir(tmp_path)  # no stray .env
    env = subprocess_env(tmp_path)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return env


@pytest.mark.parametrize("key", [None, "  "])
def test_refuses_to_start_without_an_approval_key(
    server_env: dict[str, str], monkeypatch: pytest.MonkeyPatch, key: str | None
) -> None:
    if key is None:
        monkeypatch.delenv("TW_APPROVAL_KEY")
    else:
        monkeypatch.setenv("TW_APPROVAL_KEY", key)
    with pytest.raises(SystemExit, match="TW_APPROVAL_KEY"):
        main(["--transport", "stdio"])


def test_short_key_error_does_not_echo_the_secret(
    server_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TW_APPROVAL_KEY", "short-secret-value")
    with pytest.raises(SystemExit) as info:
        main(["--transport", "http"])
    assert "at least 32" in str(info.value) and "short-secret-value" not in str(info.value)


def test_no_env_file_ignores_dotenv(
    server_env: dict[str, str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / ".env").write_text("TW_APPROVAL_KEY=short\n")
    monkeypatch.delenv("TW_APPROVAL_KEY")
    with pytest.raises(SystemExit, match="at least 32"):
        main(["--transport", "stdio"])
    monkeypatch.setenv("TW_APPROVAL_KEY", server_env["TW_APPROVAL_KEY"])
    (tmp_path / ".env").write_text("TW_LLM_PROVIDER=nonsense\n")
    ran: list[Any] = []
    monkeypatch.setattr(anyio, "run", lambda fn, server: ran.append(fn))
    main(["--transport", "stdio", "--no-env-file"])
    assert ran == [run_stdio]


def test_serves_stdio_and_http(server_env: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
    ran: list[Any] = []
    monkeypatch.setattr(anyio, "run", lambda fn, server: ran.append((fn, server)))
    main([])
    assert ran and ran[0][0] is run_stdio
    assert Path(server_env["TW_DB_PATH"]).exists()

    import uvicorn

    served: dict[str, Any] = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: served.update(app=app, **kw))
    main(["--transport", "http", "--host", "0.0.0.0", "--port", "9999"])  # noqa: S104
    assert served["host"] == "0.0.0.0" and served["port"] == 9999  # noqa: S104
    assert any(getattr(r, "path", None) == "/mcp" for r in served["app"].routes)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
        return port


@pytest.fixture
def http_server(tmp_path: Path) -> Iterator[tuple[str, Callable[[], None]]]:
    port = _free_port()

    def stop() -> None:
        proc.terminate()
        proc.wait(timeout=10)

    args = ["-m", "triagewright.mcp_server", "--transport", "http", "--port", str(port)]
    proc = subprocess.Popen(  # noqa: S603
        [sys.executable, *args],
        env={"PATH": "", **subprocess_env(tmp_path)},
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 20
        while True:
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                break
            except OSError:
                if proc.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("MCP HTTP server did not start") from None
                time.sleep(0.1)
        yield f"http://127.0.0.1:{port}/mcp", stop
    finally:
        stop()


@pytest.mark.integration
@pytest.mark.anyio
async def test_streamable_http_end_to_end(
    http_server: tuple[str, Callable[[], None]],
) -> None:
    from tests.mcp_server.test_behavior import signed_refund

    url, stop_server = http_server
    async with McpToolGateway(Client(url)) as gw:
        order = await gw.call(ToolName.GET_ORDER, {"order_id": "ord_a1001"}, ctx())
        assert order.data and order.data["order_id"] == "ord_a1001"
        foreign = await gw.call(ToolName.GET_ORDER, {"order_id": "ord_c3001"}, ctx())
        assert foreign.error and foreign.error.category is ErrorCategory.NOT_FOUND
        hal = ctx("cus_hal08", "human")
        forged = {**signed_refund(SUBPROCESS_KEY.encode()), "approval_token": "0" * 64}
        denied = await gw.call(ToolName.ISSUE_REFUND, forged, hal)
        assert denied.error and denied.error.category is ErrorCategory.PERMISSION
        args = signed_refund(SUBPROCESS_KEY.encode())
        first = await gw.call(ToolName.ISSUE_REFUND, args, hal)
        assert not first.is_error and await gw.call(ToolName.ISSUE_REFUND, args, hal) == first

        stop_server()  # a dead transport fails calls; it never cancels the caller
        with anyio.fail_after(10):
            lost = await gw.call(ToolName.GET_ORDER, {"order_id": "ord_a1001"}, ctx())
            again = await gw.call(ToolName.GET_ORDER, {"order_id": "ord_a1001"}, ctx())
        for result in (lost, again):
            assert result.error and result.error.category is ErrorCategory.TRANSIENT
            assert result.error.retryable
        assert len(gw.calls) == 7


@pytest.mark.integration
@pytest.mark.anyio
async def test_connect_failure_raises_on_enter() -> None:
    gw = McpToolGateway(Client(f"http://127.0.0.1:{_free_port()}/mcp"))
    with anyio.fail_after(10), pytest.raises(Exception):  # noqa: B017, PT011
        await gw.__aenter__()
