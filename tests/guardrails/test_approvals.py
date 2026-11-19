from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from triagewright.contracts.guardrails import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalStatus,
    GuardrailDecision,
    GuardrailVerdict,
)
from triagewright.contracts.ports import ApprovalStore
from triagewright.contracts.tools import ToolName
from triagewright.guardrails import ApprovalConflictError, SqliteApprovalStore
from triagewright.testing import seed_as_of


def request(approval_id: str = "apr_0001", amount: int = 5900) -> ApprovalRequest:
    return ApprovalRequest(
        approval_id=approval_id,
        ticket_id="tkt_0001",
        customer_id="cus_hal08",
        tool=ToolName.ISSUE_REFUND,
        arguments={"order_id": "ord_h8001", "amount_cents": amount, "reason": "damaged"},
        idempotency_key=f"refund-{approval_id}",
        rationale="damaged",
        verdict=GuardrailVerdict(
            decision=GuardrailDecision.REQUIRE_APPROVAL, rule_ids=["approval.irreversible"]
        ),
        requested_at=seed_as_of(),
    )


def decision(approved: bool = True, approval_id: str = "apr_0001") -> ApprovalDecision:
    return ApprovalDecision(
        approval_id=approval_id,
        approved=approved,
        reviewer="supervisor",
        note="checked photos",
        decided_at=seed_as_of(),
    )


def test_satisfies_the_port() -> None:
    assert isinstance(SqliteApprovalStore(), ApprovalStore)


def test_lifecycle_matches_the_fake() -> None:
    store = SqliteApprovalStore()
    req = request()
    assert store.submit(req) == req
    assert store.list_pending() == [req]
    with pytest.raises(ValueError, match="not approved"):
        store.mark_executed("apr_0001")
    decided = store.decide(decision())
    assert decided.status is ApprovalStatus.APPROVED and store.list_pending() == []
    with pytest.raises(ValueError, match="already approved"):
        store.decide(decision(approved=False))
    executed = store.mark_executed("apr_0001")
    assert executed == req.model_copy(update={"status": ApprovalStatus.EXECUTED})
    assert store.get("apr_0001") == executed and store.get("apr_nope") is None


def test_submit_is_insert_if_absent() -> None:
    store = SqliteApprovalStore()
    first = store.submit(request())
    store.decide(decision())
    again = store.submit(request(amount=1))  # same id, different body: ignored
    assert again == first.model_copy(update={"status": ApprovalStatus.APPROVED})
    assert again.arguments["amount_cents"] == 5900


def test_only_pending_requests_can_be_submitted() -> None:
    approved = request().model_copy(update={"status": ApprovalStatus.APPROVED})
    with pytest.raises(ValueError, match="only pending"):
        SqliteApprovalStore().submit(approved)


def test_conflicts_carry_the_stored_status() -> None:
    store = SqliteApprovalStore()
    store.submit(request())
    store.decide(decision(approved=False))
    with pytest.raises(ApprovalConflictError) as rejected:
        store.decide(decision())
    assert rejected.value.status is ApprovalStatus.REJECTED
    assert rejected.value.approval_id == "apr_0001"
    with pytest.raises(ApprovalConflictError, match="rejected, not approved"):
        store.mark_executed("apr_0001")


def test_unknown_ids_raise_key_error() -> None:
    store = SqliteApprovalStore()
    with pytest.raises(KeyError):
        store.decide(decision(approval_id="apr_nope"))
    with pytest.raises(KeyError):
        store.mark_executed("apr_nope")


def test_pending_list_keeps_submission_order() -> None:
    store = SqliteApprovalStore()
    ids = ["apr_ccc1", "apr_aaa1", "apr_bbb1"]
    for approval_id in ids:
        store.submit(request(approval_id))
    store.decide(decision(approval_id="apr_aaa1"))
    assert [r.approval_id for r in store.list_pending()] == ["apr_ccc1", "apr_bbb1"]


def test_state_and_audit_survive_a_restart(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "approvals.sqlite"
    store = SqliteApprovalStore(path)
    store.submit(request())
    store.decide(decision())
    store.close()
    reopened = SqliteApprovalStore(path)
    stored = reopened.get("apr_0001")
    assert stored and stored.status is ApprovalStatus.APPROVED
    reopened.close()
    with sqlite3.connect(path) as conn:
        row = conn.execute("SELECT reviewer, note, decided_at FROM approvals").fetchone()
    assert row == ("supervisor", "checked photos", seed_as_of().isoformat())


def test_a_failed_transaction_rolls_back(tmp_path: Path) -> None:
    store = SqliteApprovalStore(tmp_path / "a.sqlite")
    store.submit(request())

    def fail_midway() -> None:
        with store._transaction() as conn:
            conn.execute("UPDATE approvals SET status = 'executed'")
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        fail_midway()
    stored = store.get("apr_0001")
    assert stored and stored.status is ApprovalStatus.PENDING


def test_concurrent_decisions_have_exactly_one_winner(tmp_path: Path) -> None:
    path = tmp_path / "shared.sqlite"
    SqliteApprovalStore(path).submit(request())
    stores = [SqliteApprovalStore(path) for _ in range(8)]  # one per "process"
    wins: list[bool] = []
    barrier = threading.Barrier(len(stores))

    def run(store: SqliteApprovalStore, approved: bool) -> None:
        barrier.wait()
        try:
            store.decide(decision(approved=approved))
            wins.append(approved)
        except ApprovalConflictError:
            pass

    threads = [threading.Thread(target=run, args=(s, i % 2 == 0)) for i, s in enumerate(stores)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(wins) == 1
    stored = stores[0].get("apr_0001")
    expected = ApprovalStatus.APPROVED if wins[0] else ApprovalStatus.REJECTED
    assert stored and stored.status is expected
