"""Deterministic chunking of KB markdown on headings and paragraphs.

Each chunk is a verbatim slice of the document body, so
``doc.body[chunk.start_char:chunk.end_char] == chunk.text``, and a citation
quote checked against ``chunk.text`` is a quote from the article.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from triagewright.contracts.rag import Chunk, KBDocument

# Bump when chunk boundaries change, so re-ingest rebuilds every document.
CHUNKER_VERSION = 1
DEFAULT_MAX_CHARS = 800

_HEADING = re.compile(r"^#{1,6}[ \t]", re.MULTILINE)
_BLANK_LINE = re.compile(r"\n[ \t]*\n")

type Span = tuple[int, int]


def chunk_document(doc: KBDocument, *, max_chars: int = DEFAULT_MAX_CHARS) -> list[Chunk]:
    """Split ``doc`` into one chunk per heading section, packing paragraphs up to ``max_chars``.

    A section longer than ``max_chars`` is split between paragraphs; a single
    paragraph longer than that stays whole. Ids are ``<doc_id>#<ordinal>``.
    """
    body = doc.body
    spans = [span for section in _sections(body) for span in _pack(body, section, max_chars)]
    return [
        Chunk(
            chunk_id=f"{doc.doc_id}#{ordinal}",
            doc_id=doc.doc_id,
            ordinal=ordinal,
            title=doc.title,
            category=doc.category,
            tags=doc.tags,
            trusted=doc.trusted,
            text=body[start:end],
            start_char=start,
            end_char=end,
        )
        for ordinal, (start, end) in enumerate(spans)
    ]


def _sections(body: str) -> Iterator[Span]:
    starts = sorted({0, *(m.start() for m in _HEADING.finditer(body))})
    yield from zip(starts, [*starts[1:], len(body)], strict=True)


def _paragraphs(body: str, section: Span) -> Iterator[Span]:
    pos, end = section
    for m in [*_BLANK_LINE.finditer(body, pos, end), None]:
        stop = m.start() if m else end
        if span := _strip(body, pos, stop):
            yield span
        if m:
            pos = m.end()


def _strip(body: str, start: int, end: int) -> Span | None:
    while start < end and body[start].isspace():
        start += 1
    while end > start and body[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


def _pack(body: str, section: Span, max_chars: int) -> Iterator[Span]:
    current: Span | None = None
    for start, end in _paragraphs(body, section):
        if current and end - current[0] <= max_chars:
            current = (current[0], end)
            continue
        if current:
            yield current
        current = (start, end)
    if current:
        yield current
