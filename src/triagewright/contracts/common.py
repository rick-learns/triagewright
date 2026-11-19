"""Shared base types for every contract.

Contracts are frozen, reject unknown fields, and use integer cents for money.
Changing anything under ``triagewright.contracts`` is a contract change: it goes
through the orchestrator in a dedicated PR, and ``schemas/`` must be regenerated
with ``uv run python scripts/export_schemas.py``.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

CONTRACT_VERSION = "1.0.0"

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]


class Contract(BaseModel):
    """Base for all contract models: immutable and strict about unknown fields."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


Cents = Annotated[int, Field(ge=0, description="Amount in minor currency units (cents).")]
Currency = Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")]

CustomerId = Annotated[str, StringConstraints(pattern=r"^cus_[a-z0-9]{4,32}$")]
OrderId = Annotated[str, StringConstraints(pattern=r"^ord_[a-z0-9]{4,32}$")]
TicketId = Annotated[str, StringConstraints(pattern=r"^tkt_[a-z0-9]{4,32}$")]
RefundId = Annotated[str, StringConstraints(pattern=r"^rf_[a-z0-9]{4,32}$")]
EscalationId = Annotated[str, StringConstraints(pattern=r"^esc_[a-z0-9]{4,32}$")]
ApprovalId = Annotated[str, StringConstraints(pattern=r"^apr_[a-z0-9]{4,32}$")]
DocId = Annotated[str, StringConstraints(pattern=r"^kb_[a-z0-9_]{2,64}$")]
ChunkId = Annotated[str, StringConstraints(pattern=r"^kb_[a-z0-9_]{2,64}#\d{1,4}$")]
RequestId = Annotated[str, StringConstraints(pattern=r"^req_[a-z0-9]{8,32}$")]
