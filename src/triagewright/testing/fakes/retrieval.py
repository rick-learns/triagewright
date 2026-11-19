"""Keyword-overlap retriever and paragraph chunker (stand-ins for lane C's RAG)."""

from __future__ import annotations

import re
from collections.abc import Sequence

from triagewright.contracts.rag import Chunk, KBDocument, RetrievalQuery, RetrievedChunk

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return {t for t in _WORD.findall(text.lower()) if len(t) > 2}


def chunk_documents(docs: Sequence[KBDocument]) -> list[Chunk]:
    """Split on blank lines. A stand-in for lane C's chunker."""
    chunks: list[Chunk] = []
    for doc in docs:
        pos = 0
        ordinal = 0
        for para in doc.body.split("\n\n"):
            start = doc.body.index(para, pos)
            pos = start + len(para)
            if not para.strip() or para.lstrip().startswith("#"):
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#{ordinal}",
                    doc_id=doc.doc_id,
                    ordinal=ordinal,
                    title=doc.title,
                    category=doc.category,
                    tags=doc.tags,
                    trusted=doc.trusted,
                    text=para.strip(),
                    start_char=start,
                    end_char=pos,
                )
            )
            ordinal += 1
    return chunks


class FakeRetriever:
    """Keyword-overlap retriever over paragraph chunks."""

    def __init__(self, docs: Sequence[KBDocument]) -> None:
        self.chunks = chunk_documents(docs)

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]:
        q = _tokens(query.text)
        scored: list[tuple[float, Chunk]] = []
        for chunk in self.chunks:
            f = query.filters
            if f and f.categories and chunk.category not in f.categories:
                continue
            if f and f.tags_any and not set(f.tags_any) & set(chunk.tags):
                continue
            words = _tokens(chunk.text + " " + chunk.title)
            score = len(q & words) / (len(q) or 1)
            if score > max(query.min_score, 0.0):
                scored.append((score, chunk))
        scored.sort(key=lambda s: (-s[0], s[1].chunk_id))
        return [
            RetrievedChunk(chunk=c, score=s, rank=i + 1, method="keyword")
            for i, (s, c) in enumerate(scored[: query.k])
        ]
