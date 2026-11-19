"""Rule-based prompt-injection screening of untrusted text.

Every rule has a stable id (``injection.override``, ...). Base rules apply to
every source and catch text that tries to reprogram the assistant. Content
that reaches the model indirectly (tool output and KB text) is held to a
stricter standard: it has no reason to address an AI model or to tell the
reader to reveal, ignore, or grant anything, so the strict rules flag that
too. Rules stay narrow, because one false positive blocks every write in
the request: policy text ("never refund without approval") must pass.

A match gives ``decision=block`` and ``injection_suspected=True``: the text
is data, never instructions. The coordinator still shows it to models,
delimited, and ORs ``injection_suspected`` into the request (ADR-001 §4).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from triagewright.contracts.guardrails import ContentSource, GuardrailDecision, GuardrailVerdict

_INDIRECT = frozenset({ContentSource.TOOL_OUTPUT, ContentSource.KB_DOCUMENT})
_ALL = frozenset(ContentSource)


@dataclass(frozen=True)
class _Rule:
    rule_id: str
    reason: str
    sources: frozenset[ContentSource]
    patterns: tuple[re.Pattern[str], ...]


def _rx(*patterns: str) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(p, re.MULTILINE) for p in patterns)


_W = r"\W+"  # any run of separators
_ANY_TOOL = (
    r"(?:issue_refund|create_escalation|check_refund_eligibility|get_customer|get_account"
    r"|get_order|list_orders|search_kb)"
)
# "the previous instructions on the box" are a product's, not the assistant's. Only the
# same clause counts: spaces, not a sentence break, separate the target from the phrase.
_ON_A_THING = (
    r" (?:on|in|from|inside|with) (?:the |my |this |that |your )?(?:\w+ )?"
    r"(?:box|label|package|packaging|manual|email|letter|receipt|leaflet|insert|sticker|slip"
    r"|card|form|website|site|page|app|bottle|device)s?\b"
)
# A policy check, not a message about one ("the verification email").
_CHECK = (
    r"(?:(?:approval|verification|eligibility|policy|security|fraud)\W+"
    r"(?:checks?|requirements?|process|steps?|rules)"
    r"|(?:approval|verification)(?![ -](?:emails?|messages?|links?|codes?|texts?|letters?"
    r"|notifications?)\b))"
)
_TARGET = (
    r"(?:instructions?|rules|polic(?:y|ies)|guidelines|directives|prompts?|guardrails"
    r"|restrictions|programming|constraints|safeguards)"
)

_RULES: tuple[_Rule, ...] = (
    _Rule(
        "injection.override",
        "tries to override the assistant's instructions",
        _ALL,
        _rx(
            # The target is the assistant's instructions ("your previous rules"), not the
            # customer's own ("my earlier return instructions").
            r"\b(?:ignore|disregard|forget|override|bypass)\W+(?:(?:any|of|the|these|those)\W+){0,2}"
            r"(?:(?:all|your|previous|prior|above|earlier|preceding|system|initial|original"
            rf"|safety|developer)\W+){{1,3}}{_TARGET}\b(?!{_ON_A_THING})",
            r"\b(?:new|updated|revised|real|actual|true)\s+system\s+(?:instructions?|prompt)\b",
        ),
    ),
    _Rule(
        "injection.system_prompt",
        "asks for the system prompt or hidden instructions",
        _ALL,
        _rx(
            r"\bsystem\W+prompts?\b",
            r"\b(?:reveal|show|print|display|output|repeat|leak|dump|tell|share)\W+"
            r"(?:\w+\W+){0,3}?(?:hidden|initial|original|developer|secret|internal)\W+"
            r"(?:prompt|instructions?|rules|configuration)\b",
            r"\b(?:reveal|print|repeat|dump|leak)\W+(?:me\W+)?(?:your|the)\W+(?:\w+\W+)?"
            r"instructions\b",
        ),
    ),
    _Rule(
        "injection.role_override",
        "tries to change the assistant's role or mode",
        _ALL,
        _rx(
            r"\b(?:you\s+are|you're)\s+now\s+(?:an?\s+|the\s+|in\s+)?(?:\w+\s+)?"
            r"(?:mode|admin\w*|developer|assistant|ai|bot|model|unrestricted|jailbroken|dan"
            r"|unfiltered|root|superuser|system)\b",
            r"\bfrom\s+now\s+on,?\s+you\s+(?:are|will|must|should|shall)\b",
            r"\b(?:developer|god|admin|debug|jailbreak|dan|unrestricted|sudo)\s+mode\b",
            r"\bjailbr[eo]a?k\w*\b",
            r"\b(?:act|behave|respond)\s+as\s+(?:an?\s+|the\s+)?(?:\w+\s+)?"
            r"(?:admin\w*|developer|root|superuser|system|unrestricted|unfiltered)\b",
        ),
    ),
    _Rule(
        "injection.fake_authority",
        "impersonates a system, developer, or tool message",
        _ALL,
        _rx(
            r"^\W*(?:system|assistant|developer)\s*:",
            r"\[\s*/?\s*(?:system|admin|inst|sys)\s*\]",
            r"<\|?\s*/?\s*(?:system|im_start|im_end|endoftext|assistant)\s*\|?>",
            # A label, so only at the start of a line ("I got a system message: ..." passes).
            r"^\W*system\s+(?:note|message|notice|override|instruction|alert|directive)s?\s*:",
            r"\bsystem\s+(?:note|message|notice|override|instruction|alert|directive)s?\s+"
            r"(?:to|for)\s+(?:all\s+|any\s+|the\s+)?(?:ai|llm|assistant|bot|model)s?\b",
            r"\b(?:note|message|instructions?|notice|attention)\s+(?:to|for)\s+(?:all\s+|any\s+"
            r"|the\s+)?(?:ai|llm|chatbot|bot|assistant|language\s+model|model)s?\b",
            r"\b(?:tool_use|function_call|tool_calls?)\b",
        ),
    ),
    _Rule(
        "injection.credential_request",
        "asks for credentials or approval secrets",
        _ALL,
        _rx(
            r"\b(?:include|reveal|share|send|give|print|output|tell|disclose|show|provide|leak"
            r"|post|what\s+is|what's)\W+(?:\w+\W+){0,4}?(?:admin\w*|root|system|master|api"
            r"|internal|staff|employee|support)\W+(?:password|passwd|credentials?|keys?|tokens?"
            r"|secrets?)\b",
            r"\bapproval[\s_]+(?:token|key)s?\b",
        ),
    ),
    _Rule(
        "injection.forced_action",
        "tries to force a refund or approval past policy",
        _ALL,
        _rx(
            r"\b(?:approve|issue|grant|process|send|authori[sz]e)\W+(?:\w+\W+){0,4}?refunds?"
            r"\W+(?:to|for)\W+(?:any|every|all|each)(?:one)?\b",
            r"\bauto[- ]?approv(?:e|al)\b",
            r"\bmark\W+(?:\w+\W+){0,3}?as\W+approved\b",
        ),
    ),
    _Rule(
        "injection.tool_directive",
        "tells the assistant which tool to call",
        _ALL,
        _rx(
            rf"\b(?:call|invoke|run|use|execute|trigger)\W+(?:the\W+)?{_ANY_TOOL}\b",
            rf"\b{_ANY_TOOL}\s*\(",
        ),
    ),
    _Rule(
        "injection.addressed_to_model",
        "indirect content addresses an AI model",
        _INDIRECT,
        _rx(
            r"\b(?:dear|attention|hey|hello|hi)\W+(?:the\W+)?(?:ai|llm|assistant|bot|model)s?\b",
            r"\b(?:ai|llm)\s+(?:assistants?|agents?|models?|bots?)\s*(?::|,\s*(?:you|please)\b)",
            r"\bif\s+you\s+are\s+an?\s+(?:ai|assistant|bot|language\s+model|llm)\b",
            r"\blanguage\s+models?\s+(?:reading|processing)\s+this\b",
        ),
    ),
    _Rule(
        "injection.action_directive",
        "indirect content directs the reader past policy",
        _INDIRECT,
        _rx(
            # Only a directive that bypasses policy: "you should approve the refund only
            # after you verify eligibility" is policy, and passes.
            r"\byou\s+(?:must|should|are\s+(?:required|instructed)\s+to)\W+(?:now\W+)?"
            r"(?:reveal\W+(?:\w+\W+){0,2}?(?:password|credentials?|secrets?|keys?|tokens?"
            rf"|prompt|instructions)|(?:ignore|disregard|override|bypass)\W+(?:\w+\W+){{0,2}}?"
            rf"(?:{_TARGET}|(?:refund\W+)?limits?|{_CHECK})"
            r"|(?:grant|approve|issue)\W+(?:\w+\W+){0,2}?refunds?\W+(?:without\W+(?:\w+\W+){0,2}?"
            r"(?:approval|review|verification|authori[sz]ation|eligibility|checks?"
            r"|asking(?! \w))"  # bare "without asking", not "without asking the customer to ..."
            r"|regardless\W+of|to\W+(?:any|every|all)))\b",
        ),
    ),
)

RULE_IDS: tuple[str, ...] = tuple(r.rule_id for r in _RULES)

# Common Cyrillic and Greek look-alikes of Latin letters.
_CONFUSABLES = str.maketrans(
    "аеорсухіѕјԁɡһкмтвнΑΒΕΗΙΚΜΝΟΡΤΥΧαεικνορτυχ",
    "aeopcyxisjdghkmtbhabehikmnoptyxaeikvoptux",
)
_JSON_ESCAPES = (("\\n", "\n"), ("\\r", "\n"), ("\\t", " "), ('\\"', '"'))
_LINE_BREAKS = re.compile(r"\r\n?|[\v\f\x1c-\x1e\x85\u2028\u2029]")
_SPACES = re.compile(r"[^\S\n]+")


def normalize(text: str) -> str:
    """Fold the text so formatting tricks don't hide a match.

    NFKC (full-width and styled letters), drop format characters (zero-width
    spaces, bidi marks), map common look-alike letters, unescape the JSON
    escapes that serialized tool output carries, casefold, collapse spaces.
    """
    text = unicodedata.normalize("NFKC", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Cf")
    for escaped, plain in _JSON_ESCAPES:
        text = text.replace(escaped, plain)
    text = text.translate(_CONFUSABLES).casefold()
    return _SPACES.sub(" ", _LINE_BREAKS.sub("\n", text))  # keep line breaks for clauses


def screen(text: str, source: ContentSource) -> GuardrailVerdict:
    folded = normalize(text)
    hits = [
        rule
        for rule in _RULES
        if source in rule.sources and any(p.search(folded) for p in rule.patterns)
    ]
    if not hits:
        return GuardrailVerdict(decision=GuardrailDecision.ALLOW)
    return GuardrailVerdict(
        decision=GuardrailDecision.BLOCK,
        rule_ids=[r.rule_id for r in hits],
        reasons=[f"{source.value}: {r.reason}" for r in hits],
        injection_suspected=True,
    )
