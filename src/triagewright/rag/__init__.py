"""Lane C: KB ingest, chunking, embeddings, hybrid retrieval, citations.

Typical use (lane H wires commands on top)::

    index = KBIndex(settings.db_path, make_embedder(settings.embedder))
    index.ingest(load_kb(settings.kb_dir))  # idempotent
    retriever = HybridRetriever(index)  # a ports.Retriever

or ``build_retriever(settings)`` for both steps.
"""

from triagewright.rag.chunking import chunk_document
from triagewright.rag.citations import make_citation, verify_citations
from triagewright.rag.embedders import Embedder, FastEmbedEmbedder, HashingEmbedder, make_embedder
from triagewright.rag.index import IngestStats, KBIndex
from triagewright.rag.retriever import HybridRetriever, build_retriever

__all__ = [
    "Embedder",
    "FastEmbedEmbedder",
    "HashingEmbedder",
    "HybridRetriever",
    "IngestStats",
    "KBIndex",
    "build_retriever",
    "chunk_document",
    "make_citation",
    "make_embedder",
    "verify_citations",
]
