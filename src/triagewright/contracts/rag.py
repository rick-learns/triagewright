"""Knowledge-base documents, chunks, retrieval queries, and citations."""

from __future__ import annotations

from datetime import date
from typing import Annotated, Literal

from pydantic import Field

from triagewright.contracts.common import ChunkId, Contract, DocId


class KBDocument(Contract):
    """A KB article. Bodies are markdown and are untrusted data, never instructions."""

    doc_id: DocId
    title: str
    category: str
    tags: list[str] = []
    updated_at: date
    source: Literal["synthetic", "public"] = "synthetic"
    url: str | None = None
    body: Annotated[str, Field(min_length=1)]


class Chunk(Contract):
    chunk_id: ChunkId  # "<doc_id>#<ordinal>"
    doc_id: DocId
    ordinal: Annotated[int, Field(ge=0)]
    title: str
    category: str
    tags: list[str] = []
    text: Annotated[str, Field(min_length=1)]
    start_char: Annotated[int, Field(ge=0)]
    end_char: Annotated[int, Field(ge=0)]


class MetadataFilter(Contract):
    categories: list[str] | None = None
    tags_any: list[str] | None = None
    updated_after: date | None = None


class RetrievalQuery(Contract):
    text: Annotated[str, Field(min_length=1, max_length=500)]
    k: Annotated[int, Field(ge=1, le=20)] = 5
    filters: MetadataFilter | None = None
    min_score: float = 0.0


class RetrievedChunk(Contract):
    chunk: Chunk
    score: float
    rank: Annotated[int, Field(ge=1)]
    method: Literal["vector", "keyword", "hybrid"]


class Citation(Contract):
    """A pointer from an answer back to the chunk that supports it.

    ``quote`` must be a verbatim substring of the cited chunk's text.
    """

    doc_id: DocId
    chunk_id: ChunkId
    title: str
    quote: Annotated[str, Field(min_length=1, max_length=400)]
