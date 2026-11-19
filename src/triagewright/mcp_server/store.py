"""The mock business systems in one SQLite database.

Each table keeps the full record as JSON plus the columns the tools query by.
Records are only written from validated contract models, so the two never
disagree. ``":memory:"`` gives an isolated database (tests, per-case worlds).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Final

from pydantic import BaseModel

from triagewright.contracts.world import (
    Account,
    Customer,
    Escalation,
    Order,
    Refund,
    WorldState,
)

MEMORY: Final = ":memory:"
_SCHEMA_VERSION: Final = 1

_SCHEMA: Final = """
CREATE TABLE customers (customer_id TEXT PRIMARY KEY, data TEXT NOT NULL);
CREATE TABLE accounts (customer_id TEXT PRIMARY KEY, data TEXT NOT NULL);
CREATE TABLE orders (
    order_id TEXT PRIMARY KEY, customer_id TEXT NOT NULL, data TEXT NOT NULL
);
CREATE INDEX orders_by_customer ON orders (customer_id);
CREATE TABLE refunds (
    refund_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX refunds_by_customer ON refunds (customer_id);
CREATE INDEX refunds_by_key ON refunds (idempotency_key);
CREATE TABLE escalations (
    escalation_id TEXT PRIMARY KEY, ticket_id TEXT NOT NULL, data TEXT NOT NULL
);
CREATE INDEX escalations_by_ticket ON escalations (ticket_id);
"""


class MockSystems:
    """Customers, accounts, orders, refunds, and escalations. Insertion order is preserved."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    @classmethod
    def open(cls, path: Path | str, *, seed: WorldState) -> MockSystems:
        """Open ``path``, creating and seeding it with ``seed`` if it's new.

        An existing database keeps its state (issued refunds, escalations).
        """
        if str(path) != MEMORY:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), isolation_level=None, timeout=5.0)
        systems = cls(conn)
        with systems.transaction():
            (version,) = conn.execute("PRAGMA user_version").fetchone()
            if version == 0:
                # Not executescript: it would commit the open transaction first.
                for statement in filter(str.strip, _SCHEMA.split(";")):
                    conn.execute(statement)
                systems._load(seed)
                conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            elif version != _SCHEMA_VERSION:
                raise RuntimeError(f"{path}: unknown schema version {version}")
        return systems

    @classmethod
    def in_memory(cls, world: WorldState) -> MockSystems:
        """A fresh, isolated database holding ``world``."""
        return cls.open(MEMORY, seed=world)

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """One write transaction. ``BEGIN IMMEDIATE`` serializes writers across processes."""
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield
            self._conn.execute("COMMIT")
        except BaseException:
            if self._conn.in_transaction:  # also after a failed (busy) COMMIT
                self._conn.execute("ROLLBACK")
            raise

    # ---- reads ----

    def customer(self, customer_id: str) -> Customer | None:
        return self._one(Customer, "customers", "customer_id", customer_id)

    def account(self, customer_id: str) -> Account | None:
        return self._one(Account, "accounts", "customer_id", customer_id)

    def order(self, order_id: str) -> Order | None:
        return self._one(Order, "orders", "order_id", order_id)

    def orders_of(self, customer_id: str) -> list[Order]:
        return self._many(Order, "orders", "customer_id", customer_id)

    def refunds_of(self, customer_id: str) -> list[Refund]:
        return self._many(Refund, "refunds", "customer_id", customer_id)

    def refund_by_key(self, idempotency_key: str) -> Refund | None:
        return self._one(Refund, "refunds", "idempotency_key", idempotency_key)

    def escalation_for(self, ticket_id: str) -> Escalation | None:
        return self._one(Escalation, "escalations", "ticket_id", ticket_id)

    def ledger(self) -> WorldState:
        return WorldState(
            customers=self._all(Customer, "customers"),
            accounts=self._all(Account, "accounts"),
            orders=self._all(Order, "orders"),
            refunds=self._all(Refund, "refunds"),
            escalations=self._all(Escalation, "escalations"),
        )

    # ---- writes (call inside ``transaction()``) ----

    def add_refund(self, refund: Refund) -> None:
        self._conn.execute(
            "INSERT INTO refunds VALUES (?, ?, ?, ?, ?)",
            (
                refund.refund_id,
                refund.order_id,
                refund.customer_id,
                refund.idempotency_key,
                refund.model_dump_json(),
            ),
        )

    def add_escalation(self, escalation: Escalation) -> None:
        self._conn.execute(
            "INSERT INTO escalations VALUES (?, ?, ?)",
            (escalation.escalation_id, escalation.ticket_id, escalation.model_dump_json()),
        )

    def _load(self, world: WorldState) -> None:
        self._conn.executemany(
            "INSERT INTO customers VALUES (?, ?)",
            [(c.customer_id, c.model_dump_json()) for c in world.customers],
        )
        self._conn.executemany(
            "INSERT INTO accounts VALUES (?, ?)",
            [(a.customer_id, a.model_dump_json()) for a in world.accounts],
        )
        self._conn.executemany(
            "INSERT INTO orders VALUES (?, ?, ?)",
            [(o.order_id, o.customer_id, o.model_dump_json()) for o in world.orders],
        )
        for refund in world.refunds:
            self.add_refund(refund)
        for escalation in world.escalations:
            self.add_escalation(escalation)

    # Table and column names below are module constants, never caller input.

    def _one[M: BaseModel](self, model: type[M], table: str, column: str, value: str) -> M | None:
        row = self._conn.execute(
            f"SELECT data FROM {table} WHERE {column} = ? ORDER BY rowid LIMIT 1",  # noqa: S608
            (value,),
        ).fetchone()
        return None if row is None else model.model_validate_json(row[0])

    def _many[M: BaseModel](self, model: type[M], table: str, column: str, value: str) -> list[M]:
        rows = self._conn.execute(
            f"SELECT data FROM {table} WHERE {column} = ? ORDER BY rowid",  # noqa: S608
            (value,),
        )
        return [model.model_validate_json(data) for (data,) in rows]

    def _all[M: BaseModel](self, model: type[M], table: str) -> list[M]:
        rows = self._conn.execute(f"SELECT data FROM {table} ORDER BY rowid")  # noqa: S608
        return [model.model_validate_json(data) for (data,) in rows]
