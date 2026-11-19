from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from triagewright.contracts.rag import KBDocument
from triagewright.rag import HashingEmbedder, IngestStats, KBIndex, chunk_document
from triagewright.testing import seed_kb


def counts(path: Path) -> list[int]:
    with closing(sqlite3.connect(path)) as conn:
        return [
            conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]  # noqa: S608
            for t in ("kb_docs", "kb_chunks", "kb_chunks_fts")
        ]


def test_ingest_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "kb.sqlite"
    docs = seed_kb()
    total = sum(len(chunk_document(d)) for d in docs)
    with KBIndex(db, HashingEmbedder()) as index:
        assert index.ingest(docs) == IngestStats(added=len(docs), chunks=total)
        assert index.ingest(docs) == IngestStats(unchanged=len(docs), chunks=total)
    # A fresh process over the same file sees the stored fingerprints.
    with KBIndex(db, HashingEmbedder()) as index:
        assert index.ingest(list(reversed(docs))) == IngestStats(unchanged=len(docs), chunks=total)
    assert counts(db) == [len(docs), total, total]


def test_changed_and_removed_docs_are_replaced(kb_index: KBIndex) -> None:
    docs = seed_kb()
    shipping = next(d for d in docs if d.doc_id == "kb_shipping")
    edited = shipping.model_copy(update={"body": "## Delivery times\n\nZeppelin delivery only."})
    rest = [d for d in docs if d.doc_id not in {"kb_shipping", "kb_billing"}]
    stats = kb_index.ingest([*rest, edited])
    assert (stats.added, stats.updated, stats.unchanged, stats.removed) == (0, 1, len(rest), 1)
    assert [c.doc_id for c, _ in kb_index.keyword_search("zeppelin")] == ["kb_shipping"]
    assert kb_index.keyword_search("FedEx DHL") == []
    assert kb_index.get_chunk("kb_billing#0") is None
    assert kb_index.get_chunk("kb_shipping#1") is None


def test_get_chunk_round_trips(kb_index: KBIndex) -> None:
    expected = chunk_document(next(d for d in seed_kb() if d.doc_id == "kb_community_tips"))[0]
    assert kb_index.get_chunk(expected.chunk_id) == expected


def test_duplicate_doc_ids_are_rejected(kb_index: KBIndex) -> None:
    doc = seed_kb()[0]
    with pytest.raises(ValueError, match="duplicate"):
        kb_index.ingest([doc, doc])


def test_embedder_change_reembeds_everything(tmp_path: Path) -> None:
    db = tmp_path / "kb.sqlite"
    docs: list[KBDocument] = seed_kb()
    with KBIndex(db, HashingEmbedder()) as index:
        index.ingest(docs)
    with KBIndex(db, HashingEmbedder(dim=64)) as index:
        with pytest.raises(RuntimeError, match="re-run ingest"):
            index.vector_search("refund")
        assert index.ingest(docs).updated == len(docs)
        assert index.vector_search("refund")


def test_embeddings_are_float32_blobs(tmp_path: Path) -> None:
    db = tmp_path / "kb.sqlite"
    with KBIndex(db, HashingEmbedder(dim=128)) as index:
        index.ingest(seed_kb())
    with closing(sqlite3.connect(db)) as conn:
        sizes = {n for (n,) in conn.execute("SELECT length(embedding) FROM kb_chunks")}
    assert sizes == {128 * 4}
