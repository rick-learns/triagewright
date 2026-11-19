"""Citation checks: a citation needs a trusted chunk and a verbatim quote from its text.

Works on anything shaped like a chunk: ``rag.Chunk``, a ``RetrievedChunk.chunk``,
or a ``tools.KBHit`` from a ``search_kb`` result.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from triagewright.contracts.rag import Citation

MAX_QUOTE_CHARS = 400


class CitableSource(Protocol):
    @property
    def doc_id(self) -> str: ...

    @property
    def chunk_id(self) -> str: ...

    @property
    def title(self) -> str: ...

    @property
    def trusted(self) -> bool: ...

    @property
    def text(self) -> str: ...


def make_citation(source: CitableSource, quote: str) -> Citation | None:
    """A :class:`Citation` for ``quote`` from ``source``, or ``None`` if it isn't citable.

    ``None`` when the source is untrusted, or the quote is empty, longer than
    ``MAX_QUOTE_CHARS``, or not a verbatim substring of ``source.text``.
    """
    quote = quote.strip()
    if not source.trusted or not quote or len(quote) > MAX_QUOTE_CHARS:
        return None
    if quote not in source.text:
        return None
    return Citation(doc_id=source.doc_id, chunk_id=source.chunk_id, title=source.title, quote=quote)


def verify_citations(
    candidates: Iterable[Citation], sources: Iterable[CitableSource]
) -> list[Citation]:
    """Keep the candidates that pass :func:`make_citation` against the matching source.

    A candidate is matched by ``chunk_id`` and must name the same ``doc_id``.
    The title comes from the source, not the candidate. Unknown chunks and
    duplicates are dropped; order is preserved.
    """
    by_id = {s.chunk_id: s for s in sources}
    kept: list[Citation] = []
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        source = by_id.get(candidate.chunk_id)
        if source is None or source.doc_id != candidate.doc_id:
            continue
        citation = make_citation(source, candidate.quote)
        if citation and (citation.chunk_id, citation.quote) not in seen:
            seen.add((citation.chunk_id, citation.quote))
            kept.append(citation)
    return kept
