from __future__ import annotations

import pytest

from triagewright import api, cli, mcp_server
from triagewright.evals import main as evals_main


def test_evals_smoke_entry_point() -> None:
    assert evals_main(["smoke"]) == 0
    assert evals_main([]) == 2


@pytest.mark.parametrize("entry", [api.main, cli.main, mcp_server.main])
def test_unimplemented_entry_points_exit_cleanly(entry: object) -> None:
    with pytest.raises(SystemExit, match="not implemented"):
        entry()  # type: ignore[operator]
