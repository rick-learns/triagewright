"""The fake tool gateway: lookups, scoping, eligibility, faults, and KB search."""

from __future__ import annotations

import pytest

from tests.contracts.fakes_helpers import CTX, ctx, gateway
from triagewright.contracts.common import JsonObject
from triagewright.contracts.evals import ToolFault
from triagewright.contracts.tools import ErrorCategory, ToolError, ToolName
from triagewright.testing import FakeToolGateway, seed_as_of, seed_world

pytestmark = pytest.mark.anyio


async def test_lookups_and_not_found() -> None:
    gw = gateway()
    order = await gw.call(ToolName.GET_ORDER, {"order_id": "ord_a1001"}, CTX)
    assert not order.is_error and order.data and order.data["status"] == "delivered"
    missing = await gw.call(ToolName.GET_ORDER, {"order_id": "ord_zzzz"}, CTX)
    assert missing.error and missing.error.category is ErrorCategory.NOT_FOUND
    assert not missing.error.retryable
    lookups: list[tuple[ToolName, JsonObject, str]] = [
        (ToolName.GET_CUSTOMER, {"customer_id": "cus_ava01"}, "cus_ava01"),
        (ToolName.GET_ACCOUNT, {"customer_id": "cus_dev04"}, "cus_dev04"),
        (ToolName.LIST_ORDERS, {"customer_id": "cus_hal08", "limit": 2}, "cus_hal08"),
    ]
    for name, args, owner in lookups:
        result = await gw.call(name, args, ctx(owner))
        assert not result.is_error, result
    listing = await gw.call(ToolName.LIST_ORDERS, {"customer_id": "cus_nope1"}, ctx("cus_nope1"))
    assert listing.is_error
    assert [c.tool for c in gw.calls][:2] == [ToolName.GET_ORDER, ToolName.GET_ORDER]


async def test_records_are_scoped_to_the_ticket_customer() -> None:
    gw = gateway()
    foreign = await gw.call(ToolName.GET_ORDER, {"order_id": "ord_c3001"}, CTX)
    absent = await gw.call(ToolName.GET_ORDER, {"order_id": "ord_zzzz"}, CTX)
    assert foreign.error and absent.error
    assert foreign.error.category is absent.error.category is ErrorCategory.NOT_FOUND
    other = await gw.call(ToolName.GET_CUSTOMER, {"customer_id": "cus_cho03"}, CTX)
    assert other.error and other.error.category is ErrorCategory.NOT_FOUND
    unscoped = await gw.call(ToolName.GET_ORDER, {"order_id": "ord_a1001"}, ctx(None))
    assert unscoped.error and unscoped.error.category is ErrorCategory.PERMISSION
    kb = await gw.call(ToolName.SEARCH_KB, {"query": "refund"}, ctx(None))
    assert not kb.is_error


async def test_validation_error_for_bad_arguments() -> None:
    result = await gateway().call(ToolName.GET_ORDER, {"order_id": 7}, CTX)
    assert result.error and result.error.category is ErrorCategory.VALIDATION


async def test_refund_eligibility_uses_shared_policy() -> None:
    gw = gateway()

    async def check(order_id: str, amount: int, owner: str) -> JsonObject:
        r = await gw.call(
            ToolName.CHECK_REFUND_ELIGIBILITY,
            {"order_id": order_id, "amount_cents": amount},
            ctx(owner),
        )
        assert r.data is not None, r
        return r.data

    assert (await check("ord_h8001", 5000, "cus_hal08"))["eligible"] is True
    assert (await check("ord_b2001", 1000, "cus_ben02"))["reasons"] == ["outside_window"]
    assert (await check("ord_a1002", 100, "cus_ava01"))["reasons"] == ["not_delivered"]
    partial = await check("ord_a1001", 12900, "cus_ava01")  # 3000 already refunded
    assert partial["reasons"] == ["exceeds_balance"] and partial["max_refundable_cents"] == 9900
    kai = await check("ord_k1101", 1000, "cus_kai11")  # two refunds on other orders
    assert kai["reasons"] == ["refund_count_limit"] and kai["refunds_last_30d"] == 2
    assert (await check("ord_c3001", 89900, "cus_cho03"))["reasons"] == ["over_hard_limit"]
    ivo = await check("ord_i9001", 1200, "cus_ivo09")
    assert ivo["reasons"] == ["account_inactive", "outside_window"]


async def test_fault_injection_fails_first_n_matching_calls() -> None:
    fault = ToolFault(
        tool=ToolName.GET_ORDER,
        error=ToolError(category=ErrorCategory.TRANSIENT, message="db down", retryable=True),
        times=2,
        match_arguments={"order_id": "ord_a1001"},
    )
    gw = gateway(faults=[fault])
    other = await gw.call(ToolName.GET_ORDER, {"order_id": "ord_a1002"}, CTX)
    assert not other.is_error
    results = [await gw.call(ToolName.GET_ORDER, {"order_id": "ord_a1001"}, CTX) for _ in range(3)]
    assert [r.is_error for r in results] == [True, True, False]
    assert results[0].error and results[0].error.retryable


async def test_search_kb_marks_untrusted_docs() -> None:
    gw = gateway()
    hits = await gw.call(
        ToolName.SEARCH_KB, {"query": "refund window 30 days", "categories": ["refunds"]}, CTX
    )
    assert hits.data and hits.data["hits"]
    tips = await gw.call(ToolName.SEARCH_KB, {"query": "community tips customers", "k": 10}, CTX)
    assert tips.data and isinstance(tips.data["hits"], list)
    trusted = {h["doc_id"]: h["trusted"] for h in tips.data["hits"] if isinstance(h, dict)}
    assert trusted.get("kb_community_tips") is False
    empty = FakeToolGateway(seed_world(), as_of=seed_as_of())
    no_kb = await empty.call(ToolName.SEARCH_KB, {"query": "refund"}, CTX)
    assert no_kb.data == {"hits": []}
