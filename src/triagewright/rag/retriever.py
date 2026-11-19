"""``ports.Retriever`` over a :class:`KBIndex`: keyword, vector, or hybrid (RRF)."""

from __future__ import annotations

from typing import Literal

from triagewright.config import Settings
from triagewright.contracts.rag import RetrievalQuery, RetrievedChunk
from triagewright.data import load_kb
from triagewright.rag.embedders import make_embedder
from triagewright.rag.index import KBIndex, Scored, rank_scores

type Mode = Literal["keyword", "vector", "hybrid"]

RRF_K = 60


class HybridRetriever:
    """Ranks with BM25, cosine, or their reciprocal rank fusion.

    ``score`` is the method's own score: negated BM25, cosine, or the RRF sum
    ``Σ 1/(rrf_k + rank)``, so ``RetrievalQuery.min_score`` is per method.
    Only positive scores are returned, as in the fake.

    An untrusted chunk (``trusted=False``, user-submitted) is returned only
    when ``filters.categories`` names its category: an explicit opt-in, kept
    apart from the free-text query. Neither ranking nor query words can decide
    this: a planted page that says "issue a full refund" outranks the real
    policy for a refund about the same product, and incidental words ("user")
    match its title. Under ADR-001 one flagged hit blocks every write in the
    request. Returned untrusted chunks are screened by the caller and are
    never citable (see :mod:`triagewright.rag.citations`).
    """

    def __init__(self, index: KBIndex, *, mode: Mode = "hybrid", rrf_k: int = RRF_K) -> None:
        self.index = index
        self.mode: Mode = mode
        self.rrf_k = rrf_k

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]:
        return self.search(query)

    def close(self) -> None:
        self.index.close()

    def search(self, query: RetrievalQuery) -> list[RetrievedChunk]:
        """Synchronous :meth:`retrieve`."""
        opted_in = set(query.filters.categories or []) if query.filters else set()
        kept = [
            (c, s)
            for c, s in self._rank(query)
            if s > query.min_score and (c.trusted or c.category in opted_in)
        ][: query.k]
        return [
            RetrievedChunk(chunk=c, score=s, rank=i, method=self.mode)
            for i, (c, s) in enumerate(kept, start=1)
        ]

    def _rank(self, query: RetrievalQuery) -> Scored:
        if self.mode == "keyword":
            return self.index.keyword_search(query.text, query.filters)
        if self.mode == "vector":
            return self.index.vector_search(query.text, query.filters)
        lists = [
            self.index.keyword_search(query.text, query.filters),
            self.index.vector_search(query.text, query.filters),
        ]
        fused: dict[str, float] = {}
        chunks = {}
        for ranked in lists:
            for rank, (chunk, _) in enumerate(ranked, start=1):
                fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + 1 / (self.rrf_k + rank)
                chunks[chunk.chunk_id] = chunk
        return rank_scores([(chunks[cid], score) for cid, score in fused.items()])


def build_retriever(settings: Settings, *, mode: Mode = "hybrid") -> HybridRetriever:
    """Open the index at ``settings.db_path`` and ingest ``settings.kb_dir`` (idempotent)."""
    if not settings.kb_dir.is_dir():
        # load_kb would return [] and ingest would then empty the index.
        raise FileNotFoundError(f"KB directory not found: {settings.kb_dir}")
    index = KBIndex(settings.db_path, make_embedder(settings.embedder))
    index.ingest(load_kb(settings.kb_dir))
    return HybridRetriever(index, mode=mode)
