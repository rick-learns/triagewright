from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from triagewright.contracts.agents import ResolutionStatus, TicketCategory
from triagewright.contracts.guardrails import PIIType, PolicyConfig
from triagewright.contracts.policy import REFUND_OUTCOME, evaluate_refund
from triagewright.contracts.tickets import Ticket
from triagewright.contracts.world import WorldState
from triagewright.data import load_as_of, load_world, merge_world
from triagewright.datagen import (
    AS_OF,
    Expected,
    RefundOutcome,
    allocate,
    generate,
    parse_mix,
    words,
)
from triagewright.datagen.__main__ import main
from triagewright.guardrails import RuleGuardrails
from triagewright.testing.fixtures import REPO_ROOT, seed_as_of, seed_world

SAMPLE = REPO_ROOT / "data" / "generated" / "sample"
SAMPLE_ARGS = ["--seed", "7", "--customers", "20", "--tickets", "30"]
ADDRESS = re.compile(rf"\b\d{{1,3}} (?:{'|'.join(words.STREETS)}), (?:{'|'.join(words.TOWNS)})\b")
STREET_LIKE = re.compile(r"\b\d+ [A-Z]\w* (?:Lane|Row|Road|Court|Avenue|Crescent|Street|Way)\b")
PHONE = re.compile(r"(?:\(\d{3}\) |\d{3}-|\+1 \d{3} )555[- ]01\d\d")


def _run(out: Path, *args: str) -> dict[str, bytes]:
    assert main([*args, "--out", str(out)]) == 0
    return {p.name: p.read_bytes() for p in sorted(out.iterdir())}


def _tickets(out: Path) -> list[Ticket]:
    lines = (out / "tickets.jsonl").read_text(encoding="utf-8").splitlines()
    return [Ticket.model_validate_json(line) for line in lines]


def _expected(out: Path) -> list[Expected]:
    lines = (out / "expected.jsonl").read_text(encoding="utf-8").splitlines()
    return [Expected.model_validate_json(line) for line in lines]


@pytest.fixture(scope="module")
def out(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("gen")
    _run(path, "--seed", "11", "--customers", "40", "--tickets", "80")
    return path


def test_world_and_tickets_validate_and_load(out: Path) -> None:
    world = load_world(out)
    assert len(world.customers) == 40 and len(world.accounts) == 40
    assert load_as_of(out) == AS_OF == seed_as_of()
    merged = merge_world(seed_world(), world)
    assert len(merged.customers) == len(seed_world().customers) + 40
    tickets = _tickets(out)
    assert len(tickets) == 80
    customers = {c.customer_id for c in world.customers}
    by_id = {c.customer_id: c for c in world.customers}
    orders = {o.order_id: o for o in world.orders}
    for o in world.orders:
        assert o.placed_at > by_id[o.customer_id].created_at
    for ticket, exp in zip(tickets, _expected(out), strict=True):
        assert ticket.ticket_id == exp.ticket_id and ticket.customer_id == exp.customer_id
        assert ticket.customer_id in customers
        assert ticket.created_at > by_id[exp.customer_id].created_at
        if exp.order_id is not None:
            assert exp.order_id in ticket.body
            assert orders[exp.order_id].customer_id == ticket.customer_id


def test_same_seed_is_byte_identical(tmp_path: Path) -> None:
    first = _run(tmp_path / "a", *SAMPLE_ARGS)
    assert first == _run(tmp_path / "b", *SAMPLE_ARGS)
    assert first != _run(tmp_path / "c", "--seed", "8", *SAMPLE_ARGS[2:])


def test_committed_sample_is_current(tmp_path: Path) -> None:
    fresh = _run(tmp_path, *SAMPLE_ARGS)
    committed = {p.name: p.read_bytes() for p in sorted(SAMPLE.iterdir()) if p.name != "README.md"}
    assert committed == fresh


def test_ids_never_collide_with_the_seed(out: Path) -> None:
    world = load_world(out)
    seed = seed_world()
    taken = {c.customer_id for c in seed.customers} | {o.order_id for o in seed.orders}
    taken |= {r.refund_id for r in seed.refunds} | {e.escalation_id for e in seed.escalations}
    ours = {c.customer_id for c in world.customers} | {o.order_id for o in world.orders}
    ours |= {r.refund_id for r in world.refunds} | {t.ticket_id for t in _tickets(out)}
    assert all("_gen" in i for i in ours)
    assert not any("_gen" in i for i in taken)
    assert not ours & taken


def test_data_is_obviously_synthetic(out: Path) -> None:
    world = load_world(out)
    for c in world.customers:
        first, last = c.name.split(" ", 1)
        assert first in words.FIRST_NAMES and last in words.LAST_NAMES
        assert c.email.endswith(".invalid")
    redactor = RuleGuardrails()
    addresses = 0
    for t in _tickets(out):
        for span in redactor.redact(t.body).spans:
            value = t.body[span.start : span.end]
            if span.pii_type is PIIType.EMAIL:
                assert value.endswith(".invalid"), value
            else:
                assert span.pii_type is PIIType.PHONE and PHONE.fullmatch(value), value
        addresses += len(ADDRESS.findall(t.body))
        assert len(ADDRESS.findall(t.body)) == len(STREET_LIKE.findall(t.body))
    assert addresses > 0


def test_redactor_removes_generated_emails_and_phones(out: Path) -> None:
    redactor = RuleGuardrails()
    emails = phones = 0
    for t in _tickets(out):
        result = redactor.redact(t.body)
        types = [s.pii_type for s in result.spans]
        emails += types.count(PIIType.EMAIL)
        phones += types.count(PIIType.PHONE)
        assert ".invalid" not in result.text
        assert not PHONE.search(result.text)
        assert len(types) == len(re.findall(r"\S+@\S+\.invalid", t.body)) + len(
            PHONE.findall(t.body)
        )
    assert emails > 5 and phones > 5


def test_refund_sidecar_matches_the_policy(out: Path) -> None:
    world = merge_world(seed_world(), load_world(out))
    customers = {c.customer_id: c for c in world.customers}
    orders = {o.order_id: o for o in world.orders}
    seen: set[RefundOutcome] = set()
    for exp in _expected(out):
        if exp.refund is None:
            assert exp.category is not TicketCategory.REFUND_REQUEST
            continue
        r = exp.refund
        elig = evaluate_refund(
            order=orders[r.order_id],
            customer=customers[exp.customer_id],
            refunds=world.refunds,
            amount_cents=r.amount_cents,
            as_of=AS_OF,
            policy=PolicyConfig(),
        )
        assert elig.reasons == r.reasons, exp
        assert elig.eligible is (r.outcome is RefundOutcome.ELIGIBLE)
        if r.reasons:
            queue = REFUND_OUTCOME[r.reasons[0]]
            assert r.queue == (queue.value if queue else None)
            status = ResolutionStatus.ESCALATED if queue else ResolutionStatus.REFUSED
            assert r.status is status
        else:
            assert r.status is ResolutionStatus.PENDING_APPROVAL
        seen.add(r.outcome)
    assert seen == set(RefundOutcome)


def test_sample_covers_every_refund_outcome() -> None:
    rows = [json.loads(line) for line in (SAMPLE / "expected.jsonl").read_text().splitlines()]
    outcomes = {row["refund"]["outcome"] for row in rows if row["refund"]}
    assert outcomes == {o.value for o in RefundOutcome}


def test_mix_controls_the_category_spread(tmp_path: Path) -> None:
    gen = generate(3, 10, 12, parse_mix("refund_request=1,billing=2"))
    counts = {c: sum(e.category is c for e in gen.expected) for c in TicketCategory}
    assert counts[TicketCategory.REFUND_REQUEST] == 4 and counts[TicketCategory.BILLING] == 8
    assert sum(counts.values()) == 12
    assert allocate(10, {TicketCategory.OTHER: 1, TicketCategory.BILLING: 2}) == {
        TicketCategory.BILLING: 7,
        TicketCategory.OTHER: 3,
    }
    assert generate(3, 3, 0).tickets == []
    assert isinstance(generate(3, 3, 0).world, WorldState)


@pytest.mark.parametrize(
    ("mix", "error"),
    [
        ("refund_request", "category=weight"),
        ("nope=1", "unknown ticket category"),
        ("billing=x", "whole number"),
        ("billing=-1", "whole number"),
        ("billing=0", "not all be zero"),
    ],
)
def test_bad_mix_is_rejected(mix: str, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        parse_mix(mix)


def test_cli_rejects_bad_arguments(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["--seed", "1", "--customers", "2", "--tickets", "1", "--out", str(tmp_path)])
    with pytest.raises(SystemExit):
        main([*SAMPLE_ARGS, "--out", str(tmp_path), "--mix", "nope=1"])
    gen = generate(1, 1, 2, parse_mix("billing=1"))
    assert [e.category for e in gen.expected] == [TicketCategory.BILLING] * 2
    with pytest.raises(ValueError, match="tickets must be"):
        generate(1, 5, -1)
    with pytest.raises(ValueError, match="non-negative"):
        allocate(5, {TicketCategory.OTHER: 0})
