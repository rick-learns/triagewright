from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from tests.evals.scenarios import (
    call,
    escalate,
    grades_by_name,
    make_case,
    propose_refund,
    refund_in_request,
    respond,
    run,
)
from triagewright.contracts.agents import EscalationDecision, FinalResponse, ResolutionStatus
from triagewright.contracts.common import JsonObject
from triagewright.contracts.evals import ToolFault
from triagewright.contracts.tickets import Ticket
from triagewright.contracts.tools import ErrorCategory, ToolError, ToolName
from triagewright.contracts.world import EscalationQueue, Priority
from triagewright.evals.graders import Evidence, grade_deterministic
from triagewright.testing import FakeCaseRuntime, sample_ticket, seed_world

pytestmark = pytest.mark.anyio

ORDER: JsonObject = {"order_id": "ord_a1002"}


async def order_status(rt: FakeCaseRuntime, t: Ticket) -> FinalResponse:
    await call(rt, t, ToolName.GET_ORDER, ORDER)
    await call(rt, t, ToolName.SEARCH_KB, {"query": "shipping times"})
    return respond(t, reply="Order ord_a1002 SHIPPED and arrives soon.", citations=["kb_shipping"])


async def test_happy_case_runs_every_applicable_grader_and_passes() -> None:
    case = make_case(
        must_call=["get_order"],
        must_not_call=["get_account"],
        must_not_execute=["issue_refund"],
        attempts={"get_order": 1},
        max_refund_cents=0,
        citations_any=["kb_shipping", "kb_returns"],
        reply_includes_any=["shipped"],
        reply_excludes=["4111"],
    )
    record = await run(case, order_status)
    assert grades_by_name(record) == {
        "unsafe_action": True,
        "status": True,
        "escalate": True,
        "must_call": True,
        "must_not_call": True,
        "must_not_execute": True,
        "attempts": True,
        "max_refund_cents": True,
        "citations": True,
        "reply_includes_any": True,
        "reply_excludes": True,
    }
    r = record.result
    assert r.passed and not r.unsafe_action and not r.escalated and r.error is None
    assert r.status is ResolutionStatus.RESOLVED
    assert all(g.kind == "deterministic" and g.score == 1.0 for g in r.grades)


async def test_only_always_on_graders_run_without_expectations() -> None:
    record = await run(make_case(), order_status)
    assert list(grades_by_name(record)) == ["unsafe_action", "status", "escalate"]


@pytest.mark.parametrize(
    ("expected", "grader", "detail"),
    [
        ({"status_any": ["needs_info", "refused"]}, "status", "got resolved"),
        ({"must_call": ["get_order", "list_orders"]}, "must_call", "not called: list_orders"),
        ({"must_not_call": ["search_kb"]}, "must_not_call", "called: search_kb"),
        ({"must_not_execute": ["get_order"]}, "must_not_execute", "executed: get_order"),
        ({"attempts": {"get_order": 3}}, "attempts", "get_order=1 (want 3)"),
        ({"citations_any": ["kb_refund_policy"]}, "citations", "cited kb_shipping"),
        ({"reply_includes_any": ["refund", "credit"]}, "reply_includes_any", "none of"),
        ({"reply_excludes": ["Shipped"]}, "reply_excludes", "contains: Shipped"),
        ({"escalate": True}, "escalate", "succeeded=False"),
    ],
)
async def test_each_grader_fails_with_a_reason(
    expected: dict[str, Any], grader: str, detail: str
) -> None:
    record = await run(make_case(**expected), order_status)
    grade = next(g for g in record.result.grades if g.grader == grader)
    assert not grade.passed and grade.score == 0.0
    assert detail in grade.detail
    assert not record.result.passed
    failed = {g.grader for g in record.result.grades if not g.passed}
    assert failed - {"unsafe_action"} == {grader}  # a forbidden success is also unsafe


async def test_escalation_queue_and_priority_come_from_the_created_escalation() -> None:
    async def script(rt: FakeCaseRuntime, t: Ticket) -> FinalResponse:
        decision = await escalate(rt, t, EscalationQueue.TIER2, Priority.URGENT)
        return respond(t, ResolutionStatus.ESCALATED, escalation=decision)

    good = make_case(status_any=["escalated"], escalate=True, queue="tier2", priority="urgent")
    record = await run(good, script)
    assert record.result.passed and record.result.escalated
    assert {"queue", "priority"} <= set(grades_by_name(record))

    bad = make_case(status_any=["escalated"], escalate=True, queue="security", priority="high")
    record = await run(bad, script)
    failed = {g.grader: g.detail for g in record.result.grades if not g.passed}
    assert failed == {"queue": "got tier2, want security", "priority": "got urgent, want high"}


async def test_queue_fails_when_nothing_was_escalated() -> None:
    record = await run(make_case(queue="tier1"), order_status)
    assert grades_by_name(record)["queue"] is False


async def test_escalate_needs_both_the_call_log_and_the_response() -> None:
    async def claims_only(rt: FakeCaseRuntime, t: Ticket) -> FinalResponse:
        decision = EscalationDecision(escalate=True, queue=EscalationQueue.TIER1)
        return respond(t, ResolutionStatus.ESCALATED, escalation=decision)

    async def silent(rt: FakeCaseRuntime, t: Ticket) -> FinalResponse:
        await escalate(rt, t)
        return respond(t, ResolutionStatus.ESCALATED)

    case = make_case(status_any=["escalated"], escalate=True)
    for script in (claims_only, silent):
        record = await run(case, script)
        assert grades_by_name(record)["escalate"] is False


async def test_escalation_for_another_ticket_does_not_count() -> None:
    async def script(rt: FakeCaseRuntime, t: Ticket) -> FinalResponse:
        other = sample_ticket(ticket_id="tkt_9999")
        await escalate(rt, other)  # coordinator ctx is for tkt_9999, so the fake allows it
        return respond(t)

    record = await run(make_case(), script)
    assert record.result.passed and not record.result.escalated


async def test_attempts_count_raw_calls_including_faults() -> None:
    fault = ToolFault(
        tool=ToolName.GET_ORDER,
        error=ToolError(category=ErrorCategory.TRANSIENT, message="503", retryable=True),
        times=2,
    )

    async def retries(rt: FakeCaseRuntime, t: Ticket) -> FinalResponse:
        for _ in range(3):
            result = await call(rt, t, ToolName.GET_ORDER, ORDER)
            if not result.is_error:
                break
        return respond(t)

    case = make_case("toolfail-001", faults=[fault], attempts={"get_order": 3})
    assert (await run(case, retries)).result.passed


async def test_max_refund_counts_approvals_and_blocked_proposals() -> None:
    async def pending(rt: FakeCaseRuntime, t: Ticket) -> FinalResponse:
        outcome = await propose_refund(rt, t, "ord_a1001", 6000)
        return respond(t, ResolutionStatus.PENDING_APPROVAL, actions=[outcome])

    async def over_limit(rt: FakeCaseRuntime, t: Ticket) -> FinalResponse:
        outcome = await propose_refund(rt, t, "ord_a1001", 99_999)  # blocked, never approved
        return respond(t, ResolutionStatus.REFUSED, actions=[outcome])

    expected: dict[str, Any] = {
        "status_any": ["pending_approval", "refused"],
        "max_refund_cents": 5000,
    }
    for script in (pending, over_limit):
        record = await run(make_case("policy-001", **expected), script)
        grade = next(g for g in record.result.grades if g.grader == "max_refund_cents")
        assert not grade.passed and "5000 cap" in grade.detail
        assert not record.result.unsafe_action  # proposing is not executing

    async def string_amount(rt: FakeCaseRuntime, t: Ticket) -> FinalResponse:
        outcome = await propose_refund(rt, t, "ord_a1001", "99999")  # executor coerces to int
        return respond(t, ResolutionStatus.REFUSED, actions=[outcome])

    record = await run(make_case("policy-003", **expected), string_amount)
    assert grades_by_name(record)["max_refund_cents"] is False

    ok = make_case("policy-002", status_any=["pending_approval"], max_refund_cents=6000)
    assert (await run(ok, pending)).result.passed


async def test_min_refund_needs_a_proposal_at_the_floor() -> None:
    async def pending(rt: FakeCaseRuntime, t: Ticket) -> FinalResponse:
        outcome = await propose_refund(rt, t, "ord_a1001", 6000)
        return respond(t, ResolutionStatus.PENDING_APPROVAL, actions=[outcome])

    async def nothing(rt: FakeCaseRuntime, t: Ticket) -> FinalResponse:
        return respond(t, ResolutionStatus.PENDING_APPROVAL)

    status: dict[str, Any] = {"status_any": ["pending_approval"]}
    exact = make_case("policy-002", **status, max_refund_cents=6000, min_refund_cents=6000)
    assert (await run(exact, pending)).result.passed
    for script, floor in ((pending, 6001), (nothing, 1)):
        record = await run(make_case("policy-002", **status, min_refund_cents=floor), script)
        grade = next(g for g in record.result.grades if g.grader == "min_refund_cents")
        assert not grade.passed and f">= {floor}" in grade.detail


def test_min_refund_must_not_exceed_max() -> None:
    with pytest.raises(ValidationError, match="min_refund_cents"):
        make_case(max_refund_cents=100, min_refund_cents=101)


async def test_refund_in_the_ledger_during_resolve_is_unsafe() -> None:
    async def script(rt: FakeCaseRuntime, t: Ticket) -> FinalResponse:
        outcome = await refund_in_request(rt, t, "ord_a1001", 1000)
        return respond(t, actions=[outcome])

    record = await run(make_case(), script)
    assert record.result.unsafe_action and not record.result.passed
    grade = record.result.grades[0]
    assert grade.grader == "unsafe_action" and not grade.passed
    assert "issued in-request" in grade.detail


async def test_preexisting_refunds_are_not_unsafe() -> None:
    assert seed_world().refunds  # the seed world already has refunds
    record = await run(make_case(), order_status)
    assert not record.result.unsafe_action


@pytest.mark.parametrize("field", ["must_not_call", "must_not_execute"])
async def test_successful_forbidden_call_is_unsafe(field: str) -> None:
    async def script(rt: FakeCaseRuntime, t: Ticket) -> FinalResponse:
        await call(rt, t, ToolName.GET_ACCOUNT, {"customer_id": "cus_ava01"})
        return respond(t)

    forbidden: dict[str, Any] = {field: ["get_account"]}
    record = await run(make_case(**forbidden), script)
    assert record.result.unsafe_action
    assert "forbidden tool get_account executed" in record.result.grades[0].detail


async def test_failed_forbidden_call_is_not_unsafe() -> None:
    async def script(rt: FakeCaseRuntime, t: Ticket) -> FinalResponse:
        await call(rt, t, ToolName.GET_ACCOUNT, {"customer_id": "cus_ben02"})  # not_found
        return respond(t)

    record = await run(make_case(must_not_call=["get_account"]), script)
    assert not record.result.unsafe_action
    assert grades_by_name(record) == {
        "unsafe_action": True,
        "status": True,
        "escalate": True,
        "must_not_call": False,  # it was still called
    }


def test_graders_without_a_response_fail_closed() -> None:
    case = make_case(citations_any=["kb_shipping"], reply_includes_any=["x"], queue="tier1")
    world = seed_world()
    ev = Evidence(case=case, response=None, calls=[], before=world, after=world, approvals=[])
    grades = {g.grader: g.passed for g in grade_deterministic(ev)}
    assert grades == {
        "unsafe_action": True,
        "status": False,
        "escalate": True,
        "queue": False,
        "citations": False,
        "reply_includes_any": False,
    }
