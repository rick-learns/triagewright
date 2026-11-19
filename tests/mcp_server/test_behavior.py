"""One behavior suite, run against the fake (the reference) and every real gateway."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.mcp_server.conftest import ctx, open_harness
from triagewright.contracts.common import JsonObject
from triagewright.contracts.evals import ToolFault
from triagewright.contracts.policy import approval_token, refund_idempotency_key
from triagewright.contracts.ports import ToolGateway
from triagewright.contracts.tools import (
    TOOL_SPECS,
    CallContext,
    ErrorCategory,
    ToolError,
    ToolName,
    ToolResult,
)
from triagewright.testing import FakeRetriever, FakeToolGateway, seed_as_of, seed_kb, seed_world

pytestmark = pytest.mark.anyio

ESCALATION: JsonObject = {
    "ticket_id": "tkt_0001",
    "queue": "billing",
    "priority": "high",
    "summary": "Refund over limit",
    "reason_codes": ["policy_limit"],
}


def signed_refund(
    key: bytes,
    *,
    approval_id: str = "apr_h8001a",
    order_id: str = "ord_h8001",
    amount: int = 5900,
    idempotency_key: str | None = None,
) -> JsonObject:
    idem = idempotency_key or refund_idempotency_key(approval_id)
    return {
        "order_id": order_id,
        "amount_cents": amount,
        "reason": "damaged",
        "idempotency_key": idem,
        "approval_id": approval_id,
        "approval_token": approval_token(
            key,
            approval_id=approval_id,
            order_id=order_id,
            amount_cents=amount,
            idempotency_key=idem,
        ),
    }


def category(result: ToolResult) -> ErrorCategory | None:
    return result.error.category if result.error else None


# Every read path, compared result-for-result with the reference fake.
PARITY_CALLS: list[tuple[ToolName, JsonObject, CallContext]] = [
    (ToolName.GET_ORDER, {"order_id": "ord_a1001"}, ctx()),
    (ToolName.GET_ORDER, {"order_id": "ord_c3001"}, ctx()),
    (ToolName.GET_ORDER, {"order_id": "ord_zzzz"}, ctx()),
    (ToolName.GET_ORDER, {"order_id": 7}, ctx()),
    (ToolName.GET_ORDER, {"order_id": "ord_a1001", "extra": 1}, ctx()),
    (ToolName.GET_ORDER, {"order_id": "ord_a1001"}, ctx(None)),
    (ToolName.GET_CUSTOMER, {"customer_id": "cus_ava01"}, ctx()),
    (ToolName.GET_CUSTOMER, {"customer_id": "cus_cho03"}, ctx()),
    (ToolName.GET_ACCOUNT, {"customer_id": "cus_dev04"}, ctx("cus_dev04")),
    (ToolName.GET_ACCOUNT, {"customer_id": "cus_dev04"}, ctx()),
    (ToolName.LIST_ORDERS, {"customer_id": "cus_hal08", "limit": 2}, ctx("cus_hal08")),
    (ToolName.LIST_ORDERS, {"customer_id": "cus_hal08"}, ctx("cus_hal08")),
    (ToolName.LIST_ORDERS, {"customer_id": "cus_hal08"}, ctx()),
    (ToolName.LIST_ORDERS, {"customer_id": "cus_nope1"}, ctx("cus_nope1")),
    (ToolName.LIST_ORDERS, {"customer_id": "cus_hal08", "limit": 0}, ctx("cus_hal08")),
    (ToolName.SEARCH_KB, {"query": "refund window 30 days", "categories": ["refunds"]}, ctx()),
    (ToolName.SEARCH_KB, {"query": "community tips customers", "k": 10}, ctx(None)),
    (ToolName.SEARCH_KB, {"query": ""}, ctx()),
    (ToolName.CHECK_REFUND_ELIGIBILITY, {"order_id": "ord_h8001", "amount_cents": 5000}, ctx()),
    (
        ToolName.CHECK_REFUND_ELIGIBILITY,
        {"order_id": "ord_i9001", "amount_cents": 1200},
        ctx("cus_ivo09"),
    ),
    (
        ToolName.CHECK_REFUND_ELIGIBILITY,
        {"order_id": "ord_a1001", "amount_cents": 12900},
        ctx(),
    ),
]


async def test_reads_match_the_reference_fake(kind: str, tmp_path: Path) -> None:
    reference = FakeToolGateway(
        seed_world(), as_of=seed_as_of(), retriever=FakeRetriever(seed_kb())
    )
    async with open_harness(kind, tmp_path) as h:
        for name, args, call_ctx in PARITY_CALLS:
            expected = await reference.call(name, args, call_ctx)
            assert await h.gateway.call(name, args, call_ctx) == expected, (name, args)


async def test_lists_the_contract_tools(kind: str, tmp_path: Path) -> None:
    async with open_harness(kind, tmp_path) as h:
        assert isinstance(h.gateway, ToolGateway)
        assert list(h.gateway.list_tools()) == list(TOOL_SPECS.values())


async def test_records_are_scoped_to_the_ticket_customer(kind: str, tmp_path: Path) -> None:
    async with open_harness(kind, tmp_path) as h:
        gw = h.gateway
        foreign = await gw.call(ToolName.GET_ORDER, {"order_id": "ord_c3001"}, ctx())
        absent = await gw.call(ToolName.GET_ORDER, {"order_id": "ord_zzzz"}, ctx())
        assert category(foreign) is category(absent) is ErrorCategory.NOT_FOUND
        other = await gw.call(ToolName.GET_CUSTOMER, {"customer_id": "cus_cho03"}, ctx())
        assert category(other) is ErrorCategory.NOT_FOUND
        elig = await gw.call(
            ToolName.CHECK_REFUND_ELIGIBILITY,
            {"order_id": "ord_c3001", "amount_cents": 100},
            ctx(),
        )
        assert category(elig) is ErrorCategory.NOT_FOUND
        for name in set(TOOL_SPECS) - {ToolName.SEARCH_KB, ToolName.CREATE_ESCALATION}:
            unscoped = await gw.call(name, {}, ctx(None))
            assert category(unscoped) is ErrorCategory.PERMISSION, name
        kb = await gw.call(ToolName.SEARCH_KB, {"query": "refund"}, ctx(None))
        assert not kb.is_error


async def test_structured_errors(kind: str, tmp_path: Path) -> None:
    async with open_harness(kind, tmp_path) as h:
        bad = await h.gateway.call(ToolName.GET_ORDER, {"order_id": 7}, ctx())
        assert bad.error and bad.error.category is ErrorCategory.VALIDATION
        assert not bad.error.retryable and bad.data is None
        missing = await h.gateway.call(ToolName.GET_ORDER, {"order_id": "ord_zzzz"}, ctx())
        assert missing.error and not missing.error.retryable


async def test_escalation_is_code_owned_and_idempotent(kind: str, tmp_path: Path) -> None:
    async with open_harness(kind, tmp_path) as h:
        gw = h.gateway
        by_model = await gw.call(ToolName.CREATE_ESCALATION, ESCALATION, ctx())
        assert category(by_model) is ErrorCategory.PERMISSION
        anon = ctx(None, "coordinator")
        other = await gw.call(
            ToolName.CREATE_ESCALATION, {**ESCALATION, "ticket_id": "tkt_9999"}, anon
        )
        assert category(other) is ErrorCategory.PERMISSION
        first = await gw.call(ToolName.CREATE_ESCALATION, ESCALATION, anon)
        again = await gw.call(ToolName.CREATE_ESCALATION, {**ESCALATION, "queue": "security"}, anon)
        assert first.data and first.data["queue"] == "billing"
        assert again == first  # the ticket's one escalation wins
        assert [e.ticket_id for e in h.ledger().escalations] == ["tkt_0001"]


async def test_refund_requires_a_valid_human_approval(kind: str, tmp_path: Path) -> None:
    async with open_harness(kind, tmp_path) as h:
        gw, hal = h.gateway, ctx("cus_hal08", "human")
        forged = {**signed_refund(h.key), "approval_token": "0" * 64}
        assert category(await gw.call(ToolName.ISSUE_REFUND, forged, hal)) is (
            ErrorCategory.PERMISSION
        )
        tampered = {**signed_refund(h.key), "amount_cents": 11800}
        assert category(await gw.call(ToolName.ISSUE_REFUND, tampered, hal)) is (
            ErrorCategory.PERMISSION
        )
        by_agent = await gw.call(ToolName.ISSUE_REFUND, signed_refund(h.key), ctx("cus_hal08"))
        assert category(by_agent) is ErrorCategory.PERMISSION
        wrong_key = signed_refund(b"another-key-entirely-0123456789abcdef")
        assert category(await gw.call(ToolName.ISSUE_REFUND, wrong_key, hal)) is (
            ErrorCategory.PERMISSION
        )
        assert h.ledger().refunds == seed_world().refunds


async def test_refund_checks_ownership_then_idempotency_then_policy(
    kind: str, tmp_path: Path
) -> None:
    async with open_harness(kind, tmp_path) as h:
        gw, hal = h.gateway, ctx("cus_hal08", "human")
        args = signed_refund(h.key, amount=11800)
        foreign = await gw.call(ToolName.ISSUE_REFUND, args, ctx("cus_ava01", "human"))
        assert category(foreign) is ErrorCategory.NOT_FOUND

        committed = await gw.call(ToolName.ISSUE_REFUND, args, hal)
        assert committed.data and committed.data["idempotency_key"] == args["idempotency_key"]
        # Lost response: the same approval retried returns the committed refund,
        # even though the balance is now spent.
        retried = await gw.call(ToolName.ISSUE_REFUND, args, hal)
        assert retried == committed
        assert len(h.ledger().refunds) == len(seed_world().refunds) + 1

        reused = signed_refund(h.key, amount=100, idempotency_key=str(args["idempotency_key"]))
        assert category(await gw.call(ToolName.ISSUE_REFUND, reused, hal)) is (
            ErrorCategory.CONFLICT
        )
        # A new approval is re-checked against policy in the same transaction.
        spent = await gw.call(
            ToolName.ISSUE_REFUND, signed_refund(h.key, approval_id="apr_h8001b", amount=100), hal
        )
        assert spent.error and spent.error.category is ErrorCategory.POLICY
        assert spent.error.message == "exceeds_balance"
        old = signed_refund(h.key, approval_id="apr_b2001a", order_id="ord_b2001", amount=1000)
        outside = await gw.call(ToolName.ISSUE_REFUND, old, ctx("cus_ben02", "human"))
        assert outside.error and outside.error.message == "outside_window"
        assert len(h.ledger().refunds) == len(seed_world().refunds) + 1


async def test_faults_fail_the_first_matching_raw_calls(kind: str, tmp_path: Path) -> None:
    fault = ToolFault(
        tool=ToolName.GET_ORDER,
        error=ToolError(category=ErrorCategory.TRANSIENT, message="db down", retryable=True),
        times=2,
        match_arguments={"order_id": "ord_a1001"},
    )
    async with open_harness(kind, tmp_path, faults=[fault]) as h:
        gw = h.gateway
        other = await gw.call(ToolName.GET_ORDER, {"order_id": "ord_a1002"}, ctx())
        assert not other.is_error
        results = [
            await gw.call(ToolName.GET_ORDER, {"order_id": "ord_a1001"}, ctx()) for _ in range(3)
        ]
        assert [r.is_error for r in results] == [True, True, False]
        assert results[0].error == fault.error
        calls = h.calls()
        assert len(calls) == 4 and [c.result for c in calls[1:]] == results
        assert all(c.caller == "action" and c.tool is ToolName.GET_ORDER for c in calls)
        assert calls[0].arguments == {"order_id": "ord_a1002"}
