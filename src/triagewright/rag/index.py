"""The KB index: chunks in SQLite, FTS5 for BM25, and float32 embedding blobs.

Tables are prefixed ``kb_`` so the index can share ``Settings.db_path`` with
other lanes. Scores are rounded to 6 decimals, and ties break on
``(doc_id, ordinal)``, so ranking is identical across runs and machines.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np

from triagewright.contracts.common import canonical_json, stable_id
from triagewright.contracts.rag import Chunk, KBDocument, MetadataFilter
from triagewright.rag.chunking import CHUNKER_VERSION, DEFAULT_MAX_CHARS, chunk_document
from triagewright.rag.embedders import Embedder
from triagewright.rag.text import terms

SCORE_DECIMALS = 6

_SCHEMA = """
CREATE TABLE IF NOT EXISTS kb_docs (
    doc_id TEXT PRIMARY KEY,
    fingerprint TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS kb_chunks (
    id INTEGER PRIMARY KEY,
    chunk_id TEXT NOT NULL UNIQUE,
    doc_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    title TEXT NOT NULL,
    category TEXT NOT NULL,
    tags TEXT NOT NULL,
    trusted INTEGER NOT NULL,
    text TEXT NOT NULL,
    start_char INTEGER NOT NULL,
    end_char INTEGER NOT NULL,
    updated_at TEXT NOT NULL,
    embedder TEXT NOT NULL,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS kb_chunks_doc ON kb_chunks (doc_id);
CREATE VIRTUAL TABLE IF NOT EXISTS kb_chunks_fts
    USING fts5(title, tags, text, tokenize = 'porter unicode61');
"""

_COLUMNS = (
    "c.chunk_id, c.doc_id, c.ordinal, c.title, c.category, c.tags, c.trusted, c.text,"
    " c.start_char, c.end_char"
)

type Scored = list[tuple[Chunk, float]]


@dataclass(frozen=True)
class IngestStats:
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    chunks: int = 0  # chunks in the index after ingest


class KBIndex:
    """One SQLite-backed index bound to one embedder. Safe to share across threads."""

    def __init__(
        self,
        db_path: Path | str,
        embedder: Embedder,
        *,
        max_chars: int = DEFAULT_MAX_CHARS,
    ) -> None:
        if str(db_path) != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.max_chars = max_chars
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> KBIndex:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # ---- ingest ----------------------------------------------------------------

    def ingest(self, docs: Sequence[KBDocument]) -> IngestStats:
        """Make the index hold exactly ``docs``, in one transaction.

        Unchanged documents are skipped, changed ones are re-chunked and
        re-embedded, and documents no longer present are removed. A document
        counts as changed if its content, the chunker, or the embedder changed.
        """
        ids = [d.doc_id for d in docs]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate doc_id in ingest")
        added = updated = unchanged = 0
        with self._lock, self._conn:
            stored = dict(self._conn.execute("SELECT doc_id, fingerprint FROM kb_docs").fetchall())
            for doc in sorted(docs, key=lambda d: d.doc_id):
                fingerprint = self._fingerprint(doc)
                if stored.get(doc.doc_id) == fingerprint:
                    unchanged += 1
                    continue
                if doc.doc_id in stored:
                    self._delete(doc.doc_id)
                    updated += 1
                else:
                    added += 1
                self._insert(doc, fingerprint)
            removed = sorted(set(stored) - set(ids))
            for doc_id in removed:
                self._delete(doc_id)
            (count,) = self._conn.execute("SELECT count(*) FROM kb_chunks").fetchone()
        return IngestStats(added, updated, unchanged, len(removed), count)

    def _fingerprint(self, doc: KBDocument) -> str:
        content = canonical_json(doc.model_dump(mode="json"))
        return stable_id("kbdoc", content, CHUNKER_VERSION, self.max_chars, self.embedder.name)

    def _insert(self, doc: KBDocument, fingerprint: str) -> None:
        chunks = chunk_document(doc, max_chars=self.max_chars)
        vectors = self.embedder.embed_documents([f"{c.title}\n{c.text}" for c in chunks])
        self._conn.execute("INSERT INTO kb_docs VALUES (?, ?)", (doc.doc_id, fingerprint))
        for chunk, vector in zip(chunks, vectors, strict=True):
            tags = json.dumps(chunk.tags)
            cur = self._conn.execute(
                "INSERT INTO kb_chunks (chunk_id, doc_id, ordinal, title, category, tags, trusted,"
                " text, start_char, end_char, updated_at, embedder, embedding)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    chunk.chunk_id,
                    chunk.doc_id,
                    chunk.ordinal,
                    chunk.title,
                    chunk.category,
                    tags,
                    int(chunk.trusted),
                    chunk.text,
                    chunk.start_char,
                    chunk.end_char,
                    doc.updated_at.isoformat(),
                    self.embedder.name,
                    np.asarray(vector, dtype="<f4").tobytes(),
                ),
            )
            self._conn.execute(
                "INSERT INTO kb_chunks_fts (rowid, title, tags, text) VALUES (?, ?, ?, ?)",
                (cur.lastrowid, chunk.title, " ".join(chunk.tags), chunk.text),
            )

    def _delete(self, doc_id: str) -> None:
        self._conn.execute(
            "DELETE FROM kb_chunks_fts WHERE rowid IN (SELECT id FROM kb_chunks WHERE doc_id = ?)",
            (doc_id,),
        )
        self._conn.execute("DELETE FROM kb_chunks WHERE doc_id = ?", (doc_id,))
        self._conn.execute("DELETE FROM kb_docs WHERE doc_id = ?", (doc_id,))

    # ---- lookup and search -----------------------------------------------------

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """The stored chunk, e.g. to check a citation against the index rather than a transcript."""
        with self._lock:
            row = self._conn.execute(
                f"SELECT {_COLUMNS} FROM kb_chunks c WHERE c.chunk_id = ?",  # noqa: S608
                (chunk_id,),
            ).fetchone()
        return _chunk(row) if row else None

    def keyword_search(self, text: str, filters: MetadataFilter | None = None) -> Scored:
        """BM25 over title, tags, and text; any query term may match. Higher is better."""
        words = sorted(set(terms(text)))
        if not words:
            return []
        match = " OR ".join(f'"{w}"' for w in words)
        where, params = _where(filters)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {_COLUMNS}, bm25(kb_chunks_fts) FROM kb_chunks_fts"  # noqa: S608
                " JOIN kb_chunks c ON c.id = kb_chunks_fts.rowid"
                f" WHERE kb_chunks_fts MATCH ?{where}",
                (match, *params),
            ).fetchall()
        return rank_scores(
            [(_chunk(r[:-1]), -float(r[-1])) for r in rows if _tags_match(r[5], filters)]
        )

    def vector_search(self, text: str, filters: MetadataFilter | None = None) -> Scored:
        """Brute-force cosine against every chunk that passes ``filters``."""
        where, params = _where(filters)
        with self._lock:
            other = self._conn.execute(
                "SELECT embedder FROM kb_chunks WHERE embedder != ? LIMIT 1",
                (self.embedder.name,),
            ).fetchone()
            if other:
                raise RuntimeError(
                    f"index was built with embedder {other[0]!r}, not {self.embedder.name!r};"
                    " re-run ingest"
                )
            rows = self._conn.execute(
                f"SELECT {_COLUMNS}, c.embedding FROM kb_chunks c WHERE 1{where}",  # noqa: S608
                params,
            ).fetchall()
        rows = [r for r in rows if _tags_match(r[5], filters)]
        if not rows:
            return []
        matrix = np.stack([np.frombuffer(r[-1], dtype="<f4") for r in rows]).astype(np.float64)
        query = self.embedder.embed_query(text).astype(np.float64)
        scores = matrix @ query
        return rank_scores([(_chunk(r[:-1]), float(s)) for r, s in zip(rows, scores, strict=True)])


def _where(filters: MetadataFilter | None) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if filters and filters.categories:
        clauses.append(f"c.category IN ({', '.join('?' * len(filters.categories))})")
        params.extend(filters.categories)
    if filters and filters.updated_after:
        clauses.append("c.updated_at > ?")
        params.append(filters.updated_after.isoformat())
    return "".join(f" AND {c}" for c in clauses), params


def _tags_match(tags_json: str, filters: MetadataFilter | None) -> bool:
    if not (filters and filters.tags_any):
        return True
    return not set(filters.tags_any).isdisjoint(json.loads(tags_json))


def _chunk(row: Sequence[Any]) -> Chunk:
    chunk_id, doc_id, ordinal, title, category, tags, trusted, text, start, end = row
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        ordinal=ordinal,
        title=title,
        category=category,
        tags=json.loads(tags),
        trusted=bool(trusted),
        text=text,
        start_char=start,
        end_char=end,
    )


def rank_scores(scored: Scored) -> Scored:
    """Positive scores only, rounded, best first, ties on ``(doc_id, ordinal)``."""
    rounded = [(c, round(s, SCORE_DECIMALS)) for c, s in scored]
    kept = [(c, s) for c, s in rounded if s > 0]
    return sorted(kept, key=lambda cs: (-cs[1], cs[0].doc_id, cs[0].ordinal))
