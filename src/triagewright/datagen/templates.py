"""Ticket text: subject/body templates per category, contact lines, and money formatting."""

from __future__ import annotations

import random

from triagewright.contracts.agents import TicketCategory


def _money(cents: int) -> str:
    return f"${cents // 100}.{cents % 100:02d}"


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
        ("Can't sign in", "I can't log in to my account. Can you help?"),
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
        ("Poor service", "I've had a frustrating experience with order {order_id}."),
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
