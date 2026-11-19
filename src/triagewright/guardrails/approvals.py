"""SQLite-backed :class:`~triagewright.contracts.ports.ApprovalStore`."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from triagewright.contracts.guardrails import ApprovalDecision, ApprovalRequest, ApprovalStatus

_SCHEMA = """
CREATE TABLE IF NOT EXISTS approvals (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    approval_id TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL,
    request_json TEXT NOT NULL,
    reviewer TEXT,
    note TEXT,
    decided_at TEXT
)
"""


class ApprovalConflictError(ValueError):
    """A decision or state change that the approval's current status doesn't allow.

    ``status`` is the stored status, so the API can answer 409 and say why
    (for example, a repeat approve of an ``executed`` request).
    """

    def __init__(self, approval_id: str, status: ApprovalStatus, message: str) -> None:
        super().__init__(message)
        self.approval_id = approval_id
        self.status = status


class SqliteApprovalStore:
    """Approvals in one SQLite table. Every transition is a guarded single-row update.

    ``decide`` only moves ``pending`` to ``approved``/``rejected``, and
    ``mark_executed`` only moves ``approved`` to ``executed``; anything else
    raises :class:`ApprovalConflictError` (a ``ValueError``), and an unknown
    id raises ``KeyError``. The decision's reviewer, note, and time are kept
    for audit. Safe across threads, and across processes on one file.
    """

    def __init__(self, path: Path | str = ":memory:", *, timeout_s: float = 5.0) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(path), isolation_level=None, check_same_thread=False, timeout=timeout_s
        )
        self._conn.execute(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
                self._conn.execute("COMMIT")  # can fail on a busy database: roll back too
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    @staticmethod
    def _load(row: tuple[str, str] | None) -> ApprovalRequest | None:
        if row is None:
            return None
        request_json, status = row
        stored = ApprovalRequest.model_validate_json(request_json)
        return stored.model_copy(update={"status": ApprovalStatus(status)})

    def _get(self, conn: sqlite3.Connection, approval_id: str) -> ApprovalRequest | None:
        row = conn.execute(
            "SELECT request_json, status FROM approvals WHERE approval_id = ?", (approval_id,)
        ).fetchone()
        return self._load(row)

    def submit(self, request: ApprovalRequest) -> ApprovalRequest:
        """Insert if absent. If ``approval_id`` exists, return the stored request unchanged.

        A new request must be ``pending``: nothing skips the human.
        """
        with self._transaction() as conn:
            if (stored := self._get(conn, request.approval_id)) is not None:
                return stored
            if request.status is not ApprovalStatus.PENDING:
                raise ValueError(f"{request.approval_id}: only pending requests can be submitted")
            conn.execute(
                "INSERT INTO approvals (approval_id, status, request_json) VALUES (?, ?, ?)",
                (request.approval_id, request.status.value, request.model_dump_json()),
            )
            stored = self._get(conn, request.approval_id)
        assert stored is not None
        return stored

    def get(self, approval_id: str) -> ApprovalRequest | None:
        with self._lock:
            return self._get(self._conn, approval_id)

    def _transition(
        self,
        approval_id: str,
        *,
        expected: ApprovalStatus,
        to: ApprovalStatus,
        decision: ApprovalDecision | None = None,
    ) -> ApprovalRequest:
        with self._transaction() as conn:
            current = self._get(conn, approval_id)
            if current is None:
                raise KeyError(approval_id)
            if current.status is not expected:
                if expected is ApprovalStatus.PENDING:
                    message = f"{approval_id} is already {current.status.value}"
                else:
                    message = f"{approval_id} is {current.status.value}, not {expected.value}"
                raise ApprovalConflictError(approval_id, current.status, message)
            if decision is None:
                conn.execute(
                    "UPDATE approvals SET status = ? WHERE approval_id = ?",
                    (to.value, approval_id),
                )
            else:
                conn.execute(
                    "UPDATE approvals SET status = ?, reviewer = ?, note = ?, decided_at = ?"
                    " WHERE approval_id = ?",
                    (
                        to.value,
                        decision.reviewer,
                        decision.note,
                        decision.decided_at.isoformat(),
                        approval_id,
                    ),
                )
        return current.model_copy(update={"status": to})

    def decide(self, decision: ApprovalDecision) -> ApprovalRequest:
        to = ApprovalStatus.APPROVED if decision.approved else ApprovalStatus.REJECTED
        return self._transition(
            decision.approval_id, expected=ApprovalStatus.PENDING, to=to, decision=decision
        )

    def mark_executed(self, approval_id: str) -> ApprovalRequest:
        return self._transition(
            approval_id, expected=ApprovalStatus.APPROVED, to=ApprovalStatus.EXECUTED
        )

    def list_pending(self) -> list[ApprovalRequest]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT request_json, status FROM approvals WHERE status = ? ORDER BY seq",
                (ApprovalStatus.PENDING.value,),
            ).fetchall()
        return [r for row in rows if (r := self._load(row)) is not None]
