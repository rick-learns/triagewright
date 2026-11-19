from __future__ import annotations

import random

import pytest

from triagewright.contracts.common import canonical_json
from triagewright.contracts.guardrails import PIIType
from triagewright.guardrails import RuleGuardrails
from triagewright.guardrails.redaction import luhn_valid
from triagewright.testing import seed_world

G = RuleGuardrails()

# (text, expected redacted text, span types in order)
CORPUS: list[tuple[str, str, list[PIIType]]] = [
    ("card 4111 1111 1111 1111 please", "card [CARD_NUMBER_1] please", [PIIType.CARD_NUMBER]),
    ("card 4111-1111-1111-1111.", "card [CARD_NUMBER_1].", [PIIType.CARD_NUMBER]),
    ("amex 378282246310005", "amex [CARD_NUMBER_1]", [PIIType.CARD_NUMBER]),
    ("card 4111 1111 1111 1111 12/28", "card [CARD_NUMBER_1] 12/28", [PIIType.CARD_NUMBER]),
    ("card 4111\u00a01111\u00a01111\u00a01111", "card [CARD_NUMBER_1]", [PIIType.CARD_NUMBER]),
    (
        "cards 4111 1111 1111 1111 5555 5555 5555 4444",
        "cards [CARD_NUMBER_1] [CARD_NUMBER_2]",
        [PIIType.CARD_NUMBER, PIIType.CARD_NUMBER],
    ),
    (
        "cards 3782 822463 10005 4242 4242 4242 4242",
        "cards [CARD_NUMBER_1] [CARD_NUMBER_2]",
        [PIIType.CARD_NUMBER, PIIType.CARD_NUMBER],
    ),
    ("card 4111 1111 1111 1111 1228 123", "card [CARD_NUMBER_1] 1228 123", [PIIType.CARD_NUMBER]),
    ("card 6 4111 1111 1111 1111", "card 6 [CARD_NUMBER_1]", [PIIType.CARD_NUMBER]),
    ("card 411-1111-1111-11111", "card [CARD_NUMBER_1]", [PIIType.CARD_NUMBER]),
    ("ssn 123-45-6789", "ssn [SSN_1]", [PIIType.SSN]),
    ("ssn 123 45 6789", "ssn [SSN_1]", [PIIType.SSN]),
    ("my SSN is 123456789", "my SSN is [SSN_1]", [PIIType.SSN]),
    ("mail Ann.Lee+x@mail.example.com now", "mail [EMAIL_1] now", [PIIType.EMAIL]),
    ("mail _name@example.com", "mail [EMAIL_1]", [PIIType.EMAIL]),
    ("mail +tag@example.com", "mail [EMAIL_1]", [PIIType.EMAIL]),
    ("mail 'name@example.com'", "mail '[EMAIL_1]'", [PIIType.EMAIL]),
    ("mail o'connor@example.com", "mail [EMAIL_1]", [PIIType.EMAIL]),
    ("call (555) 123-4567", "call [PHONE_1]", [PIIType.PHONE]),
    ("call 555.123.4567 or +1 555 987 6543", "call [PHONE_1] or [PHONE_2]", [PIIType.PHONE] * 2),
    ("uk +44 20 7946 0958", "uk [PHONE_1]", [PIIType.PHONE]),
    ("from 203.0.113.42.", "from [IP_ADDRESS_1].", [PIIType.IP_ADDRESS]),
    ("v6 2001:db8::1 here", "v6 [IP_ADDRESS_1] here", [PIIType.IP_ADDRESS]),
    ("my verification code is 482913", "my verification code is [ONE_TIME_CODE_1]", []),
    ("OTP: 1234", "OTP: [ONE_TIME_CODE_1]", []),
    ("the code they texted me was 77812", "the code they texted me was [ONE_TIME_CODE_1]", []),
    ("2FA code 004321 didn't work", "2FA code [ONE_TIME_CODE_1] didn't work", []),
    ("123456 is your verification code", "[ONE_TIME_CODE_1] is your verification code", []),
    ("My OTP is 123 456", "My OTP is [ONE_TIME_CODE_1]", []),
    ('{"otp":"123456"}', '{"otp":"[ONE_TIME_CODE_1]"}', []),
    ("OTP **123456**", "OTP **[ONE_TIME_CODE_1]**", []),
    ("OTP `123456`", "OTP `[ONE_TIME_CODE_1]`", []),
    ("verification code (123456)", "verification code ([ONE_TIME_CODE_1])", []),
]

# Lookalikes that must stay as they are.
NEGATIVES = [
    "card 4111 1111 1111 1112 failed",  # fails Luhn
    "ssn 000-12-3456 and 666-12-3456 and 900-12-3456",  # never-issued SSN areas
    "Where is my order ord_a1001? It was $129.00 on 2026-06-15.",
    "tracking 1Z1001012, 12900 cents, refund rf_k1101a, 3 items",
    "zip code 94110 and promo code 2024",
    "firmware v1.2.3.4, time 10:30:45, note::",
    "ticket tkt_0001 for cus_ava01 at 12:00",
    "My login code for order 123456 never arrived",
    "Tracking number: 5551234567, tracking #555-123-4567",
    '{"carrier":"ups","tracking_number":"5551234567"}',
]


@pytest.mark.parametrize(("text", "expected", "types"), CORPUS)
def test_redaction_corpus(text: str, expected: str, types: list[PIIType]) -> None:
    result = G.redact(text)
    assert result.text == expected
    assert [s.pii_type for s in result.spans] == types


@pytest.mark.parametrize("text", NEGATIVES)
def test_lookalikes_are_not_redacted(text: str) -> None:
    result = G.redact(text)
    assert result.text == text and result.spans == []


def test_spans_point_into_the_original_text() -> None:
    text = "a@example.com then 4111111111111111 then a@example.com and b@example.com"
    result = G.redact(text)
    assert result.text == (
        "[EMAIL_1] then [CARD_NUMBER_1] then [EMAIL_1] and [EMAIL_2]"
    )  # the same value keeps its placeholder
    originals = [text[s.start : s.end] for s in result.spans]
    assert originals == ["a@example.com", "4111111111111111", "a@example.com", "b@example.com"]
    assert [s.placeholder for s in result.spans] == [
        "[EMAIL_1]",
        "[CARD_NUMBER_1]",
        "[EMAIL_1]",
        "[EMAIL_2]",
    ]


def test_redaction_is_deterministic_and_idempotent() -> None:
    text = "Email A@Example.com or a@example.com, card 5555 5555 5555 4444"
    first = G.redact(text)
    assert first == G.redact(text)
    assert first.text == "Email [EMAIL_1] or [EMAIL_1], card [CARD_NUMBER_1]"  # case-folded
    assert G.redact(first.text).text == first.text


def test_numbering_continues_after_existing_placeholders() -> None:
    reply = "We emailed [EMAIL_1]. Also reach us at help@example.com."
    assert G.redact(reply).text == "We emailed [EMAIL_1]. Also reach us at [EMAIL_2]."


def test_a_card_is_never_also_a_phone() -> None:
    result = G.redact("4111 1111 1111 1111")
    assert [s.pii_type for s in result.spans] == [PIIType.CARD_NUMBER]


def _luhn_number(rng: random.Random, length: int) -> str:
    body = "".join(rng.choice("0123456789") for _ in range(length - 1))
    return next(body + d for d in "0123456789" if luhn_valid(body + d))


def test_no_luhn_valid_number_survives_verbatim() -> None:
    rng = random.Random(7)
    for _ in range(2000):
        digits = _luhn_number(rng, rng.randint(13, 19))
        groups, rest = [], digits
        while rest:
            size = rng.randint(1, 6)
            groups.append(rest[:size])
            rest = rest[size:]
        number = "".join(g + rng.choice([" ", "-", "\u00a0"]) for g in groups)[:-1]
        text = G.redact(f"pay with {number} today").text
        assert number not in text, (number, text)


def test_luhn() -> None:
    assert luhn_valid("4111111111111111") and luhn_valid("79927398713")
    assert not luhn_valid("4111111111111112") and not luhn_valid("")


def test_seed_records_keep_everything_but_emails() -> None:
    world = seed_world()
    for order in world.orders:
        assert G.redact(canonical_json(order.model_dump(mode="json"))).spans == []
    for customer in world.customers:
        spans = G.redact(canonical_json(customer.model_dump(mode="json"))).spans
        assert [s.pii_type for s in spans] == [PIIType.EMAIL]
