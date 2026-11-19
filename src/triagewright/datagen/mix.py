"""Ticket category weights: the default mix, parsing ``--mix``, and splitting a total."""

from __future__ import annotations

from collections.abc import Mapping

from triagewright.contracts.agents import TicketCategory

DEFAULT_MIX: dict[TicketCategory, int] = {
    c: 3 if c is TicketCategory.REFUND_REQUEST else 1 for c in TicketCategory
}


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
