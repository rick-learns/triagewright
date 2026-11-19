from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from triagewright.contracts.world import Refund, WorldState
from triagewright.mcp_server.store import MockSystems
from triagewright.testing import seed_as_of, seed_world


def _refund() -> Refund:
    return Refund(
        refund_id="rf_test0001",
        order_id="ord_h8001",
        customer_id="cus_hal08",
        amount_cents=100,
        reason="x",
        idempotency_key="refund-apr_test",
        created_at=seed_as_of(),
    )


def test_in_memory_round_trips_the_world() -> None:
    systems = MockSystems.in_memory(seed_world())
    assert systems.ledger() == seed_world()
    assert MockSystems.in_memory(WorldState()).ledger() == WorldState()


def test_file_database_is_seeded_once_and_persists(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "mock.sqlite"
    first = MockSystems.open(path, seed=seed_world())
    with first.transaction():
        first.add_refund(_refund())
    first.close()
    reopened = MockSystems.open(path, seed=WorldState())  # seed ignored: not new
    assert reopened.ledger().refunds == [*seed_world().refunds, _refund()]
    assert reopened.refund_by_key("refund-apr_test") == _refund()


def test_transaction_rolls_back_on_error() -> None:
    systems = MockSystems.in_memory(seed_world())
    with pytest.raises(sqlite3.IntegrityError), systems.transaction():  # noqa: PT012
        systems.add_refund(_refund())
        systems.add_refund(_refund())  # duplicate primary key
    assert systems.ledger().refunds == seed_world().refunds


def test_failed_commit_rolls_back_and_frees_the_connection(tmp_path: Path) -> None:
    path = tmp_path / "mock.sqlite"
    MockSystems.open(path, seed=seed_world()).close()
    writer = MockSystems(sqlite3.connect(path, isolation_level=None, timeout=0.05))
    reader = sqlite3.connect(path, isolation_level=None)
    reader.execute("BEGIN")
    reader.execute("SELECT * FROM refunds").fetchall()  # a shared lock blocks COMMIT
    with pytest.raises(sqlite3.OperationalError, match="locked"), writer.transaction():
        writer.add_refund(_refund())
    reader.execute("COMMIT")
    with writer.transaction():  # not stuck inside the failed transaction
        writer.add_refund(_refund())
    assert writer.ledger().refunds == [*seed_world().refunds, _refund()]


def test_unknown_schema_version_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "mock.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA user_version = 99")
    conn.close()
    with pytest.raises(RuntimeError, match="schema version 99"):
        MockSystems.open(path, seed=seed_world())
