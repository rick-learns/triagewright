"""Seeded synthetic world and ticket generator.

Everything comes from ``random.Random(seed)`` and the fixed lists in
:mod:`triagewright.datagen.words`, so a seed always gives the same records.
Ids use a ``gen`` marker (``cus_gen00001``, ``ord_gen000001``, ``rf_gen000001``,
``tkt_gen00001``), which no seed id has. Refund tickets rotate through every
refund policy outcome; the intended outcome of each ticket goes into an
:class:`Expected` row for drafting eval cases.

:func:`generate` lives here; the pieces it uses are in ``models`` (clock, policy,
output records), ``mix`` (category weights), ``builder`` (the world), ``refunds``
(refund scenarios), ``templates`` (ticket text), and ``writer`` (:func:`write`).
Every name this module defined before the split stays importable from it.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import timedelta
from typing import Any

from triagewright.contracts.agents import TicketCategory
from triagewright.contracts.tickets import Channel, Ticket
from triagewright.contracts.world import CustomerStatus, OrderStatus, WorldState
from triagewright.datagen import words as words
from triagewright.datagen.builder import _Builder as _Builder
from triagewright.datagen.builder import _roles as _roles
from triagewright.datagen.mix import DEFAULT_MIX as DEFAULT_MIX
from triagewright.datagen.mix import allocate as allocate
from triagewright.datagen.mix import parse_mix as parse_mix
from triagewright.datagen.models import AS_OF as AS_OF
from triagewright.datagen.models import POLICY as POLICY
from triagewright.datagen.models import Expected as Expected
from triagewright.datagen.models import ExpectedRefund as ExpectedRefund
from triagewright.datagen.models import Generated as Generated
from triagewright.datagen.models import RefundOutcome as RefundOutcome
from triagewright.datagen.refunds import _expected_refund as _expected_refund
from triagewright.datagen.refunds import _refund_scenario as _refund_scenario
from triagewright.datagen.templates import _TEMPLATES as _TEMPLATES
from triagewright.datagen.templates import _contact as _contact
from triagewright.datagen.templates import _money as _money
from triagewright.datagen.writer import _json as _json
from triagewright.datagen.writer import write as write


def generate(
    seed: int, customers: int, tickets: int, mix: Mapping[TicketCategory, int] | None = None
) -> Generated:
    """Build a world of ``customers`` customers and ``tickets`` tickets from ``seed``."""
    if tickets < 0:
        raise ValueError("tickets must be >= 0")
    mix = dict(mix or DEFAULT_MIX)
    counts = allocate(tickets, mix)
    if customers < 1:
        raise ValueError("need at least 1 customer")
    if counts.get(TicketCategory.REFUND_REQUEST) and customers < 3:
        raise ValueError("refund tickets need 3 customers (normal, inactive, frequent refunder)")
    b = _Builder(seed)
    rng = b.rng
    inactive_n, frequent_n = _roles(rng, customers)
    inactive: list[str] = []
    frequent: list[str] = []
    normal: list[str] = []
    for n in range(1, customers + 1):
        if n in inactive_n:
            status = rng.choice([CustomerStatus.SUSPENDED, CustomerStatus.CLOSED])
        else:
            status = CustomerStatus.ACTIVE
        cid = b.customer(n, status).customer_id
        (inactive if n in inactive_n else frequent if n in frequent_n else normal).append(cid)
        for _ in range(rng.randint(1, 2)):
            b.order(cid, rng.choice(list(OrderStatus)))
        if n in frequent_n:
            for _ in range(POLICY.max_refunds_per_customer_30d):
                days = rng.randint(10, POLICY.refund_window_days - 5)
                b.refund(b.order(cid, OrderStatus.DELIVERED, delivered_days_ago=days))

    categories = [c for c, k in counts.items() for _ in range(k)]
    rng.shuffle(categories)
    outcomes = list(RefundOutcome)
    rng.shuffle(outcomes)
    by_id = {c.customer_id: c for c in b.customers}
    out_tickets: list[Ticket] = []
    expected: list[Expected] = []
    refund_i = 0
    for i, category in enumerate(categories, start=1):
        tid = f"tkt_gen{i:05d}"
        refund: ExpectedRefund | None = None
        fields = {
            "address": f"{rng.randint(1, 240)} {rng.choice(words.STREETS)}, "
            f"{rng.choice(words.TOWNS)}",
            "reason": rng.choice(words.REFUND_REASONS),
        }
        if category is TicketCategory.REFUND_REQUEST:
            outcome = outcomes[refund_i % len(outcomes)]
            refund_i += 1
            cid, order, amount = _refund_scenario(b, outcome, normal, inactive, frequent)
            refund = _expected_refund(order, amount, outcome)
            fields["amount"] = _money(amount)
            if outcome is RefundOutcome.NOT_DELIVERED:
                fields["reason"] = "I changed my mind and don't want it anymore"
        else:
            cid = rng.choice([c.customer_id for c in b.customers])
            order = rng.choice(b.orders_by_customer[cid])
        cust = by_id[cid]
        subject, body = rng.choice(_TEMPLATES[category])
        fields |= {
            "order_id": order.order_id,
            "item": order.items[0].name,
            "first": cust.name.split()[0],
        }
        text = body.format(**fields) + _contact(rng, cust.email, b.phones[cid])
        mentions_order = "{order_id}" in body
        out_tickets.append(
            Ticket(
                ticket_id=tid,
                customer_id=cid,
                channel=rng.choice(list(Channel)),
                subject=subject.format(**fields),
                body=f"{text}\n\nThanks,\n{cust.name}",
                created_at=AS_OF - timedelta(minutes=rng.randint(5, 24 * 60)),
                metadata={"source": "triagewright.datagen"},
            )
        )
        expected.append(
            Expected(
                ticket_id=tid,
                category=category,
                customer_id=cid,
                order_id=order.order_id if mentions_order else None,
                refund=refund,
            )
        )

    world = WorldState(
        customers=b.customers, accounts=b.accounts, orders=b.orders, refunds=b.refunds
    )
    meta: dict[str, Any] = {
        "as_of": AS_OF.isoformat().replace("+00:00", "Z"),
        "generator": "triagewright.datagen",
        "seed": seed,
        "customers": customers,
        "tickets": tickets,
        "mix": {c.value: w for c, w in sorted(mix.items()) if w > 0},
        "note": "Synthetic data. Invented names, .invalid emails, 555-01xx phones.",
    }
    return Generated(world=world, tickets=out_tickets, expected=expected, meta=meta)
