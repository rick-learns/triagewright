"""Seeded synthetic world and ticket generator.

Everything comes from ``random.Random(seed)`` and the fixed lists in
:mod:`triagewright.datagen.words`, so a seed always gives the same records.
Ids use a ``gen`` marker (``cus_gen00001``, ``ord_gen000001``, ``rf_gen000001``,
``tkt_gen00001``), which no seed id has. Refund tickets rotate through every
refund policy outcome; the intended outcome of each ticket goes into an
:class:`Expected` row for drafting eval cases.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from triagewright.contracts.agents import ResolutionStatus, TicketCategory
from triagewright.contracts.common import canonical_json
from triagewright.contracts.guardrails import PolicyConfig
from triagewright.contracts.policy import REFUND_OUTCOME
from triagewright.contracts.tickets import Channel, Ticket
from triagewright.contracts.tools import RefundReason
from triagewright.contracts.world import (
    Account,
    Customer,
    CustomerStatus,
    CustomerTier,
    Order,
    OrderItem,
    OrderStatus,
    Refund,
    WorldState,
)
from triagewright.datagen import words

# The seed world's clock (data/seed/meta.json), so generated records merge with it.
AS_OF = datetime(2026, 6, 15, 12, tzinfo=UTC)
# Generated refunds are built for the default policy ($500 cap, 30 days, 2 per 30 days).
POLICY = PolicyConfig()


class RefundOutcome(StrEnum):
    """What a generated refund request is built to hit: eligible, or one policy reason."""

    ELIGIBLE = "eligible"
    ACCOUNT_INACTIVE = RefundReason.ACCOUNT_INACTIVE.value
    NOT_DELIVERED = RefundReason.NOT_DELIVERED.value
    OUTSIDE_WINDOW = RefundReason.OUTSIDE_WINDOW.value
    EXCEEDS_BALANCE = RefundReason.EXCEEDS_BALANCE.value
    OVER_HARD_LIMIT = RefundReason.OVER_HARD_LIMIT.value
    REFUND_COUNT_LIMIT = RefundReason.REFUND_COUNT_LIMIT.value


DEFAULT_MIX: dict[TicketCategory, int] = {
    c: 3 if c is TicketCategory.REFUND_REQUEST else 1 for c in TicketCategory
}


class ExpectedRefund(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    order_id: str
    amount_cents: int
    outcome: RefundOutcome
    reasons: list[RefundReason]  # what evaluate_refund should return, in precedence order
    status: ResolutionStatus  # per docs/contracts/behavior.md "Refund decisions"
    queue: str | None


class Expected(BaseModel):
    """One ``expected.jsonl`` row: what a generated ticket was built to exercise."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ticket_id: str
    category: TicketCategory
    customer_id: str
    order_id: str | None
    refund: ExpectedRefund | None = None


@dataclass
class Generated:
    world: WorldState
    tickets: list[Ticket]
    expected: list[Expected]
    meta: dict[str, Any] = field(default_factory=dict)


def parse_mix(text: str) -> dict[TicketCategory, int]:
    """Parse ``refund_request=3,billing=1`` into category weights."""
    mix: dict[TicketCategory, int] = {}
    for part in text.split(","):
        name, sep, weight = part.strip().partition("=")
        if not sep:
            raise ValueError(f"mix entry {part!r} must look like category=weight")
        try:
            category = TicketCategory(name.strip())
        except ValueError:
            raise ValueError(f"unknown ticket category {name.strip()!r}") from None
        if not weight.strip().isdigit():
            raise ValueError(f"weight for {category.value} must be a whole number")
        mix[category] = int(weight)
    if sum(mix.values()) <= 0:
        raise ValueError("mix weights must not all be zero")
    return mix


def allocate(total: int, mix: Mapping[TicketCategory, int]) -> dict[TicketCategory, int]:
    """Split ``total`` tickets by weight (largest remainder, ties in enum order)."""
    weight_sum = sum(mix.values())
    if weight_sum <= 0 or any(w < 0 for w in mix.values()):
        raise ValueError("mix weights must be non-negative and not all zero")
    ordered = [c for c in TicketCategory if mix.get(c, 0) > 0]
    counts = {c: total * mix[c] // weight_sum for c in ordered}
    by_remainder = sorted(ordered, key=lambda c: -(total * mix[c] % weight_sum))
    for c in by_remainder[: total - sum(counts.values())]:
        counts[c] += 1
    return counts


def _money(cents: int) -> str:
    return f"${cents // 100}.{cents % 100:02d}"


class _Builder:
    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)  # noqa: S311 - reproducible fixtures, not secrets
        self.customers: list[Customer] = []
        self.accounts: list[Account] = []
        self.orders: list[Order] = []
        self.refunds: list[Refund] = []
        self.phones: dict[str, str] = {}
        self.orders_by_customer: dict[str, list[Order]] = {}

    def _at(self, days_ago: int, hours: int = 0) -> datetime:
        return AS_OF - timedelta(days=days_ago, hours=hours)

    def customer(self, n: int, status: CustomerStatus) -> Customer:
        rng = self.rng
        first, last = rng.choice(words.FIRST_NAMES), rng.choice(words.LAST_NAMES)
        local = f"{first}.{last}.{n}".lower().replace("-", "")
        cid = f"cus_gen{n:05d}"
        cust = Customer(
            customer_id=cid,
            name=f"{first} {last}",
            email=f"{local}@{rng.choice(words.EMAIL_DOMAINS)}",
            tier=rng.choice([CustomerTier.STANDARD] * 3 + [CustomerTier.PLUS, CustomerTier.VIP]),
            status=status,
            region=rng.choice(words.REGIONS),
            created_at=self._at(rng.randint(60, 900)),
        )
        locked = rng.random() < 0.15
        self.accounts.append(
            Account(
                customer_id=cid,
                email_verified=rng.random() < 0.9,
                two_factor_enabled=rng.random() < 0.4,
                locked=locked,
                failed_login_attempts=rng.randint(5, 9) if locked else rng.randint(0, 2),
                last_login_at=self._at(rng.randint(0, 30), rng.randint(0, 23)),
                plan=rng.choice(["basic", "basic", "plus"]),
            )
        )
        area = rng.choice(words.AREA_CODES)
        line = f"01{rng.randint(0, 99):02d}"
        self.phones[cid] = rng.choice(
            [f"({area}) 555-{line}", f"{area}-555-{line}", f"+1 {area} 555 {line}"]
        )
        self.customers.append(cust)
        self.orders_by_customer[cid] = []
        return cust

    def order(
        self,
        customer_id: str,
        status: OrderStatus,
        *,
        delivered_days_ago: int | None = None,
        premium: bool = False,
    ) -> Order:
        rng = self.rng
        sku, name, price = rng.choice(words.PREMIUM_ITEMS if premium else words.REGULAR_ITEMS)
        items = [OrderItem(sku=sku, name=name, quantity=1, unit_price_cents=price)]
        delivered = status in {OrderStatus.DELIVERED, OrderStatus.RETURNED}
        if delivered:
            days = delivered_days_ago if delivered_days_ago is not None else rng.randint(2, 120)
            delivered_at: datetime | None = self._at(days)
            placed_at = self._at(days + rng.randint(2, 6))
        else:
            delivered_at = None
            placed_at = self._at(rng.randint(1, 5), rng.randint(0, 23))
        n = len(self.orders) + 1
        shipped = status is not OrderStatus.PLACED and status is not OrderStatus.CANCELLED
        order = Order(
            order_id=f"ord_gen{n:06d}",
            customer_id=customer_id,
            status=status,
            items=items,
            total_cents=price,
            placed_at=placed_at,
            delivered_at=delivered_at,
            carrier=rng.choice(words.CARRIERS) if shipped else None,
            tracking_number=f"SYNTRK{n:08d}" if shipped else None,
        )
        self.orders.append(order)
        self.orders_by_customer[customer_id].append(order)
        return order

    def refund(self, order: Order) -> Refund:
        """An issued refund within the last 30 days on a delivered ``order``."""
        assert order.delivered_at is not None
        n = len(self.refunds) + 1
        days_ago = (AS_OF - order.delivered_at).days
        refund = Refund(
            refund_id=f"rf_gen{n:06d}",
            order_id=order.order_id,
            customer_id=order.customer_id,
            amount_cents=order.total_cents,
            reason=self.rng.choice(words.REFUND_REASONS),
            idempotency_key=f"datagen-rf-gen{n:06d}",
            created_at=self._at(self.rng.randint(0, days_ago - 1), self.rng.randint(1, 6)),
        )
        self.refunds.append(refund)
        return refund


def _roles(rng: random.Random, n: int) -> tuple[set[int], set[int]]:
    """Customer numbers that are inactive, and that already have 2 refunds this month."""
    numbers = list(range(1, n + 1))
    rng.shuffle(numbers)
    k = max(1, n // 10)
    return set(numbers[:k]), set(numbers[k : 2 * k])


def _refund_scenario(
    b: _Builder,
    outcome: RefundOutcome,
    normal: Sequence[str],
    inactive: Sequence[str],
    frequent: Sequence[str],
) -> tuple[str, Order, int]:
    """Build a fresh order (and pick its customer) so exactly ``outcome`` applies."""
    rng = b.rng
    window = POLICY.refund_window_days
    pool = {RefundOutcome.ACCOUNT_INACTIVE: inactive, RefundOutcome.REFUND_COUNT_LIMIT: frequent}
    cid = rng.choice(pool.get(outcome, normal))
    recent = rng.randint(1, window - 2)
    match outcome:
        case RefundOutcome.NOT_DELIVERED:
            order = b.order(cid, rng.choice([OrderStatus.PLACED, OrderStatus.SHIPPED]))
        case RefundOutcome.OUTSIDE_WINDOW:
            order = b.order(cid, OrderStatus.DELIVERED, delivered_days_ago=window + recent)
        case RefundOutcome.OVER_HARD_LIMIT:
            order = b.order(cid, OrderStatus.DELIVERED, delivered_days_ago=recent, premium=True)
        case _:
            order = b.order(cid, OrderStatus.DELIVERED, delivered_days_ago=recent)
    amount = order.total_cents
    if outcome is RefundOutcome.EXCEEDS_BALANCE:
        amount += rng.randint(5, 30) * 100
    elif outcome is RefundOutcome.ELIGIBLE and rng.random() < 0.3:
        amount = max(100, amount // 2)
    return cid, order, amount


def _expected_refund(order: Order, amount: int, outcome: RefundOutcome) -> ExpectedRefund:
    if outcome is RefundOutcome.ELIGIBLE:
        return ExpectedRefund(
            order_id=order.order_id,
            amount_cents=amount,
            outcome=outcome,
            reasons=[],
            status=ResolutionStatus.PENDING_APPROVAL,
            queue=None,
        )
    reason = RefundReason(outcome.value)
    queue = REFUND_OUTCOME[reason]
    return ExpectedRefund(
        order_id=order.order_id,
        amount_cents=amount,
        outcome=outcome,
        reasons=[reason],
        status=ResolutionStatus.REFUSED if queue is None else ResolutionStatus.ESCALATED,
        queue=None if queue is None else queue.value,
    )


# (subject, body) per category. Fields: {order_id}, {item}, {address}, {first}.
_TEMPLATES: dict[TicketCategory, tuple[tuple[str, str], ...]] = {
    TicketCategory.ORDER_STATUS: (
        ("Where is my order?", "Hi, can you tell me the status of order {order_id}?"),
        ("Order update", "I placed order {order_id} for the {item}. Has it shipped yet?"),
    ),
    TicketCategory.SHIPPING: (
        ("Shipping address", "Please ship order {order_id} to {address} instead."),
        ("Delivery time", "How long does delivery take to {address}? It's for {order_id}."),
    ),
    TicketCategory.REFUND_REQUEST: (
        (
            "Refund request",
            "Hi, I'd like a refund of {amount} for order {order_id} ({item}). {reason}.",
        ),
        ("Money back please", "{reason}. Please refund {amount} on order {order_id}."),
    ),
    TicketCategory.RETURN: (
        ("Return", "How do I return the {item} from order {order_id}?"),
        ("Send it back", "I want to send back order {order_id}. Do you send a return label?"),
    ),
    TicketCategory.BILLING: (
        ("Charged twice", "I think I was charged twice for order {order_id}. Please check."),
        ("Invoice", "Can I get an invoice for order {order_id}? My billing address is {address}."),
    ),
    TicketCategory.ACCOUNT_ACCESS: (
        ("Can't sign in", "I can't log in to my account. It says it is locked."),
        ("Password reset", "The password reset email never arrives. What should I do?"),
    ),
    TicketCategory.PRODUCT_QUESTION: (
        ("Question about the {item}", "Does the {item} work with a laptop? I have {order_id}."),
        ("Product care", "How should I clean the {item}?"),
    ),
    TicketCategory.POLICY_QUESTION: (
        ("Refund policy", "How many days do I have to ask for a refund?"),
        ("Warranty", "What does the warranty cover, and for how long?"),
    ),
    TicketCategory.COMPLAINT: (
        ("Not happy", "The {item} from {order_id} feels cheap. I expected better."),
        ("Poor service", "Your courier left order {order_id} in the rain. Very disappointed."),
    ),
    TicketCategory.OTHER: (
        ("Hello", "Hi, {first} here. Do you have a newsletter I can join?"),
        ("Gift wrap", "Do you offer gift wrapping?"),
    ),
}


def _contact(rng: random.Random, email: str, phone: str) -> str:
    match rng.choice(["none", "none", "email", "phone", "both"]):
        case "email":
            return f" You can email me at {email}."
        case "phone":
            return f" Call me on {phone} if that's easier."
        case "both":
            return f" Reach me at {email} or {phone}."
    return ""


def generate(
    seed: int, customers: int, tickets: int, mix: Mapping[TicketCategory, int] | None = None
) -> Generated:
    """Build a world of ``customers`` customers and ``tickets`` tickets from ``seed``."""
    if customers < 3:
        raise ValueError("need at least 3 customers (normal, inactive, and frequent refunders)")
    if tickets < 0:
        raise ValueError("tickets must be >= 0")
    mix = dict(mix or DEFAULT_MIX)
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

    counts = allocate(tickets, mix)
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


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def write(gen: Generated, out: Path) -> list[Path]:
    """Write the world files, ``meta.json``, ``tickets.jsonl``, and ``expected.jsonl``."""
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {
        f"{name}.json": _json([r.model_dump(mode="json") for r in getattr(gen.world, name)])
        for name in WorldState.model_fields
    }
    files["meta.json"] = _json(gen.meta)
    files["tickets.jsonl"] = "".join(
        canonical_json(t.model_dump(mode="json")) + "\n" for t in gen.tickets
    )
    files["expected.jsonl"] = "".join(
        canonical_json(e.model_dump(mode="json")) + "\n" for e in gen.expected
    )
    paths = []
    for name, text in sorted(files.items()):
        path = out / name
        path.write_text(text, encoding="utf-8")
        paths.append(path)
    return paths
