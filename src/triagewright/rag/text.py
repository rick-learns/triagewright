"""Term extraction shared by the keyword query builder and the hashing embedder."""

from __future__ import annotations

import re

_WORD = re.compile(r"[^\W_]+")

# Small English list: these words carry no topic signal in support questions.
STOPWORDS = frozenset(
    """
    a about an and are as at be been but by can could did do does for from had has have how
    i if in into is it its me my of on or our so than that the their them then there these
    they this to us was we were what when where which who why will with would you your
    """.split()  # noqa: SIM905
)


def terms(text: str) -> list[str]:
    """Lowercased word tokens without stopwords, in order (duplicates kept)."""
    return [t for t in _WORD.findall(text.casefold()) if t not in STOPWORDS]
