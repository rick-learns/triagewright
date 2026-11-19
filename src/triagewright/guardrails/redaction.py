"""PII redaction with deterministic, numbered placeholders.

Detectors run in priority order (card, SSN, one-time code, email, phone, IP)
and a later detector never claims text an earlier one already matched. Each
distinct value gets one placeholder (``[EMAIL_1]``, ``[CARD_NUMBER_2]``, ...),
numbered by first appearance, so the same input always redacts the same way.
Numbering continues after placeholders already in the text, so re-redacting
a reply never reuses a number for a different value.

One-time codes are redacted from the text but have no ``PIIType`` yet, so
they get no :class:`RedactionSpan`.
"""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from triagewright.contracts.guardrails import PIIType, RedactionResult, RedactionSpan


@dataclass(frozen=True)
class _Kind:
    label: str  # placeholder prefix
    pii_type: PIIType | None  # None: redacted, but not reportable as a span yet


_CARD = _Kind("CARD_NUMBER", PIIType.CARD_NUMBER)
_SSN = _Kind("SSN", PIIType.SSN)
_CODE = _Kind("ONE_TIME_CODE", None)
_EMAIL = _Kind("EMAIL", PIIType.EMAIL)
_PHONE = _Kind("PHONE", PIIType.PHONE)
_IP = _Kind("IP_ADDRESS", PIIType.IP_ADDRESS)


@dataclass(frozen=True)
class _Match:
    kind: _Kind
    start: int
    end: int
    key: str  # normalized value: equal keys share a placeholder


def _digits(text: str) -> str:
    return "".join(c for c in text if c.isdigit())


def luhn_valid(number: str) -> bool:
    """Luhn checksum over the digits of ``number``."""
    digits = [int(c) for c in reversed(_digits(number))]
    if not digits:
        return False
    total = 0
    for i, d in enumerate(digits):
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# A run of digit groups split by single spaces (incl. no-break and thin) or hyphens.
_DIGIT_RUN = re.compile(r"(?<![\w-])\d+(?:[ \u00a0\u2009\u202f-]\d+)*(?![A-Za-z_])")
_GROUP = re.compile(r"\d+")


def _cards(text: str) -> Iterator[_Match]:
    """Luhn-valid 13-19 digit windows of whole groups, taken left to right.

    A card starts at the first group that can start one; of the windows from
    there, the one that leaves the most digits coverable after it wins (ties:
    the longer card). So "4111 1111 1111 1111 1228 123" keeps the expiry and
    CVV, and two cards in a row are both redacted, whatever their group lengths.
    """
    for run in _DIGIT_RUN.finditer(text):
        groups = list(_GROUP.finditer(text, run.start(), run.end()))
        n = len(groups)
        # covered[i]: digits this scan covers in groups[i:]; card_end[i]: exclusive end of
        # the card it takes at groups[i], or 0 if no card starts there.
        covered, card_end = [0] * (n + 1), [0] * (n + 1)
        for i in reversed(range(n)):
            best, digits = -1, ""
            for j in range(i, n):
                digits += groups[j].group()
                if len(digits) > 19:
                    break
                total = len(digits) + covered[j + 1]
                if len(digits) >= 13 and total >= best and luhn_valid(digits):
                    best, card_end[i] = total, j + 1
            covered[i] = best if card_end[i] else covered[i + 1]
        i = 0
        while i < n:
            if not card_end[i]:
                i += 1
                continue
            digits = "".join(g.group() for g in groups[i : card_end[i]])
            yield _Match(_CARD, groups[i].start(), groups[card_end[i] - 1].end(), digits)
            i = card_end[i]


_SSN_RE = re.compile(r"(?<![\w-])(\d{3})([- ])(\d{2})\2(\d{4})(?![\w-])")
_SSN_LABELED_RE = re.compile(
    r"\b(?:ssn|social\s+security(?:\s+(?:number|no\.?|#))?)\s*(?:is|was|:|#)?\s*(\d{9})\b",
    re.IGNORECASE,
)


def _ssn_valid(digits: str) -> bool:
    area, group, serial = digits[:3], digits[3:5], digits[5:]
    return area not in {"000", "666"} and area[0] != "9" and group != "00" and serial != "0000"


def _ssns(text: str) -> Iterator[_Match]:
    for m in _SSN_RE.finditer(text):
        digits = m.group(1) + m.group(3) + m.group(4)
        if _ssn_valid(digits):
            yield _Match(_SSN, m.start(), m.end(), digits)
    for m in _SSN_LABELED_RE.finditer(text):
        if _ssn_valid(m.group(1)):
            yield _Match(_SSN, m.start(1), m.end(1), m.group(1))


_QUALIFIED = (
    r"(?:one[- ]time|verification|security|login|log[- ]in|sign[- ]in|auth|authentication"
    r"|2fa|mfa|sms|text|confirmation|access|reset|backup)\s+(?:pass)?codes?"
)
_CODE_DIGITS = r"(?<![\w-])(?P<code>\d{3}[ -]\d{3}|\d{4,8})(?![\w-])"
# Only these words may sit between a label and its code ("the code they texted me was").
_GAP_WORDS = (
    r"is|was|were|are|of|number|no|they|you|we|it|that|sent|texted|emailed|gave|me|i|got"
    r"|received|just|today"
)
_CODE_RE = re.compile(
    rf"\b(?P<label>(?P<qual>{_QUALIFIED})|otp|passcode|pin|codes?)\b"
    rf"(?P<gap>(?:[\s:#=,.\"'`*()\[\]-]|\b(?:{_GAP_WORDS})\b)*?){_CODE_DIGITS}",
    re.IGNORECASE,
)
# "123456 is your verification code"
_CODE_FIRST_RE = re.compile(
    rf"{_CODE_DIGITS}\s+(?:is|was)\s+(?:your|my|the)\s+(?:{_QUALIFIED}|otp|passcode|code)\b",
    re.IGNORECASE,
)
# A bare "code" preceded by one of these is not a one-time code ("zip code 94110").
_NOT_A_CODE = frozenset(
    {"zip", "postal", "promo", "discount", "coupon", "gift", "error", "product", "tracking"}
    | {"order", "area", "country", "status", "source", "tax", "item", "sku", "voucher"}
)
_WORD_BEFORE = re.compile(r"([a-z]+)\W*$", re.IGNORECASE)


def _codes(text: str) -> Iterator[_Match]:
    for m in _CODE_RE.finditer(text):
        if m.group("qual") is None and m.group("label").lower().startswith("code"):
            before = _WORD_BEFORE.search(text, 0, m.start())
            if before and before.group(1).lower() in _NOT_A_CODE:
                continue
        yield _Match(_CODE, m.start("code"), m.end("code"), _digits(m.group("code")))
    for m in _CODE_FIRST_RE.finditer(text):
        yield _Match(_CODE, m.start("code"), m.end("code"), _digits(m.group("code")))


# RFC 5322 atext. A leading apostrophe or backtick is a quote around the address.
_LOCAL_CHARS = r"A-Za-z0-9!#$%&'*+/=?^_`{|}~\-"
_LOCAL_FIRST = r"A-Za-z0-9!#$%&*+/=?^_{|}~\-"
_EMAIL_RE = re.compile(
    rf"(?<![.{_LOCAL_FIRST}])[{_LOCAL_FIRST}](?:[{_LOCAL_CHARS}.]{{0,62}}[{_LOCAL_CHARS}])?"
    r"@(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,24}(?![\w-])"
)


def _emails(text: str) -> Iterator[_Match]:
    for m in _EMAIL_RE.finditer(text):
        yield _Match(_EMAIL, m.start(), m.end(), m.group().lower())


_PHONE_RES = (
    # North American: (555) 123-4567, 555.123.4567, +1 555 123 4567, 5551234567.
    re.compile(
        r"(?<![\w+])(?:\+?1[\s.-]?)?(?:\(\d{3}\)\s?|\d{3}[\s.-]?)\d{3}[\s.-]?\d{4}(?![\w-])"
    ),
    # International E.164-style: +44 20 7946 0958 (8-15 digits after the +).
    re.compile(r"(?<![\w+])\+\d(?:[\s.-]?\(?\d\)?){7,14}(?![\w-])"),
)


# A carrier's tracking number is order data the reply needs, not a phone number.
_TRACKING_LABEL = re.compile(
    r"tracking(?:[\s_-]*(?:number|num|no\.?|#|id))?[\"']?\s*(?:[:=]|is)?\s*[\"']?$",
    re.IGNORECASE,
)


def _phones(text: str) -> Iterator[_Match]:
    for pattern in _PHONE_RES:
        for m in pattern.finditer(text):
            if _TRACKING_LABEL.search(text, max(0, m.start() - 40), m.start()):
                continue
            yield _Match(_PHONE, m.start(), m.end(), _digits(m.group()))


_IPV4_RE = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?!\w|\.\w)")
_IPV6_RE = re.compile(r"(?<![\w:.])[0-9A-Fa-f]{0,4}(?::[0-9A-Fa-f]{0,4}){2,7}(?![\w:])")


def _ips(text: str) -> Iterator[_Match]:
    for pattern in (_IPV4_RE, _IPV6_RE):
        for m in pattern.finditer(text):
            if not any(c.isalnum() for c in m.group()):
                continue  # a bare "::" in prose
            try:
                addr = ipaddress.ip_address(m.group())
            except ValueError:
                continue
            yield _Match(_IP, m.start(), m.end(), addr.compressed)


_DETECTORS: tuple[Callable[[str], Iterator[_Match]], ...] = (
    _cards,
    _ssns,
    _codes,
    _emails,
    _phones,
    _ips,
)
_ALL_KINDS = (_CARD, _SSN, _CODE, _EMAIL, _PHONE, _IP)
_EXISTING = re.compile(r"\[(" + "|".join(k.label for k in _ALL_KINDS) + r")_(\d+)\]")


def _select(text: str) -> list[_Match]:
    """Matches from every detector, earlier detectors first, without overlaps."""
    taken: list[_Match] = []
    for detect in _DETECTORS:
        for m in detect(text):
            if all(m.end <= t.start or m.start >= t.end for t in taken):
                taken.append(m)
    return sorted(taken, key=lambda m: m.start)


def redact(text: str) -> RedactionResult:
    matches = _select(text)
    if not matches:
        return RedactionResult(text=text)
    counters: dict[str, int] = {}
    for existing in _EXISTING.finditer(text):
        label, n = existing.group(1), int(existing.group(2))
        counters[label] = max(counters.get(label, 0), n)
    placeholders: dict[tuple[str, str], str] = {}
    spans: list[RedactionSpan] = []
    parts: list[str] = []
    pos = 0
    for m in matches:
        ident = (m.kind.label, m.key)
        if ident not in placeholders:
            counters[m.kind.label] = counters.get(m.kind.label, 0) + 1
            placeholders[ident] = f"[{m.kind.label}_{counters[m.kind.label]}]"
        placeholder = placeholders[ident]
        parts += [text[pos : m.start], placeholder]
        pos = m.end
        if m.kind.pii_type is not None:
            spans.append(
                RedactionSpan(
                    pii_type=m.kind.pii_type, start=m.start, end=m.end, placeholder=placeholder
                )
            )
    parts.append(text[pos:])
    return RedactionResult(text="".join(parts), spans=spans)
