"""Inbound support tickets."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated

from pydantic import Field

from triagewright.contracts.common import Contract, CustomerId, TicketId


class Channel(StrEnum):
    EMAIL = "email"
    CHAT = "chat"
    WEB = "web"


class Ticket(Contract):
    """A customer message as it arrives. ``body`` is untrusted customer content."""

    ticket_id: TicketId
    customer_id: CustomerId | None = None
    channel: Channel = Channel.EMAIL
    subject: Annotated[str, Field(max_length=300)] = ""
    body: Annotated[str, Field(min_length=1, max_length=20_000)]
    created_at: datetime
    metadata: dict[str, str] = {}
