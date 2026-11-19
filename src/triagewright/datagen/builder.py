"""The seeded world builder: customers, accounts, orders, and issued refunds."""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from triagewright.contracts.world import (
    Account,
    Customer,
    CustomerStatus,
    CustomerTier,
    Order,
    OrderItem,
    OrderStatus,
    Refund,
)
from triagewright.datagen import words
from triagewright.datagen.models import AS_OF


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
            created_at=self._at(rng.randint(150, 900)),
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
