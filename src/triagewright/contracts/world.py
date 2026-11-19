"""Mock business-system records (customers, accounts, orders, refunds, escalations).

All data is synthetic. Emails use reserved domains (``example.com``/``.test``)
and names are invented.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated

from pydantic import Field, model_validator

from triagewright.contracts.common import (
    Cents,
    Contract,
    Currency,
    CustomerId,
    EscalationId,
    OrderId,
    RefundId,
    TicketId,
)


class CustomerTier(StrEnum):
    STANDARD = "standard"
    PLUS = "plus"
    VIP = "vip"


class CustomerStatus(StrEnum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    CLOSED = "closed"


class Customer(Contract):
    customer_id: CustomerId
    name: str
    email: str
    tier: CustomerTier = CustomerTier.STANDARD
    status: CustomerStatus = CustomerStatus.ACTIVE
    region: Annotated[str, Field(pattern=r"^[A-Z]{2}$")] = "US"
    created_at: datetime


class Account(Contract):
    customer_id: CustomerId
    email_verified: bool = True
    two_factor_enabled: bool = False
    locked: bool = False
    failed_login_attempts: Annotated[int, Field(ge=0)] = 0
    last_login_at: datetime | None = None
    plan: str = "basic"


class OrderStatus(StrEnum):
    PLACED = "placed"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    RETURNED = "returned"


class OrderItem(Contract):
    sku: str
    name: str
    quantity: Annotated[int, Field(ge=1)]
    unit_price_cents: Cents


class Order(Contract):
    order_id: OrderId
    customer_id: CustomerId
    status: OrderStatus
    items: Annotated[list[OrderItem], Field(min_length=1)]
    total_cents: Cents
    currency: Currency = "USD"
    placed_at: datetime
    delivered_at: datetime | None = None
    carrier: str | None = None
    tracking_number: str | None = None

    @model_validator(mode="after")
    def _total_matches_items(self) -> Order:
        expected = sum(item.quantity * item.unit_price_cents for item in self.items)
        if expected != self.total_cents:
            raise ValueError(f"total_cents {self.total_cents} != sum of items {expected}")
        return self


class RefundStatus(StrEnum):
    ISSUED = "issued"
    REJECTED = "rejected"


class Refund(Contract):
    refund_id: RefundId
    order_id: OrderId
    customer_id: CustomerId
    amount_cents: Annotated[int, Field(gt=0)]
    currency: Currency = "USD"
    reason: str
    status: RefundStatus = RefundStatus.ISSUED
    idempotency_key: str
    approval_id: str | None = None
    created_at: datetime


class EscalationQueue(StrEnum):
    TIER1 = "tier1"
    TIER2 = "tier2"
    BILLING = "billing"
    SECURITY = "security"


class Priority(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class Escalation(Contract):
    escalation_id: EscalationId
    ticket_id: TicketId
    queue: EscalationQueue
    priority: Priority
    summary: str
    reason_codes: list[str]
    status: str = "open"
    created_at: datetime


class WorldState(Contract):
    """A snapshot of every mock system. Seed data and eval-case overrides use this shape.

    Eval cases merge their ``world`` over the seed world by primary key.
    """

    customers: list[Customer] = []
    accounts: list[Account] = []
    orders: list[Order] = []
    refunds: list[Refund] = []
    escalations: list[Escalation] = []
