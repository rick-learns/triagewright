from __future__ import annotations

import sys
import types
from collections.abc import Iterator, Sequence

import numpy as np
import pytest

from triagewright.contracts.rag import RetrievalQuery
from triagewright.rag import (
    FastEmbedEmbedder,
    HashingEmbedder,
    HybridRetriever,
    KBIndex,
    make_embedder,
)
from triagewright.testing import seed_kb


def test_hashing_vectors_are_deterministic_and_normalized() -> None:
    texts = ["Refunds within 30 days", "the and of", "Refunded within thirty days"]
    a = HashingEmbedder().embed_documents(texts)
    b = HashingEmbedder().embed_documents(texts)
    assert a.dtype == np.float32 and a.shape == (3, 512)
    assert a.tobytes() == b.tobytes()
    assert np.allclose(np.linalg.norm(a, axis=1), [1.0, 0.0, 1.0], atol=1e-6)
    assert float(a[0] @ a[2]) > 0.3  # trigrams relate refund/refunded
    assert np.array_equal(HashingEmbedder().embed_query(texts[0]), a[0])


def test_hashing_embedder_validates_dim() -> None:
    assert HashingEmbedder(dim=64).name == "hashing-v1-64"
    with pytest.raises(ValueError, match="dim"):
        HashingEmbedder(dim=4)


def test_make_embedder_defaults_to_hashing() -> None:
    assert isinstance(make_embedder("hashing"), HashingEmbedder)


def test_fastembed_missing_gives_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "fastembed", None)
    with pytest.raises(RuntimeError, match="local-embeddings"):
        make_embedder("fastembed")


class _FakeTextEmbedding:
    """Stands in for ``fastembed.TextEmbedding``: vectors from the hashing embedder."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._inner = HashingEmbedder(dim=32)

    def passage_embed(self, texts: Sequence[str]) -> Iterator[np.ndarray]:
        yield from 3.0 * self._inner.embed_documents(texts)

    def query_embed(self, texts: Sequence[str]) -> Iterator[np.ndarray]:
        yield from 2.0 * self._inner.embed_documents(texts)


def test_fastembed_embedder_is_lazy_and_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("fastembed")
    module.TextEmbedding = _FakeTextEmbedding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", module)
    embedder = make_embedder("fastembed")
    assert isinstance(embedder, FastEmbedEmbedder)
    assert embedder.name == "fastembed:BAAI/bge-small-en-v1.5"
    docs = embedder.embed_documents(["refund policy", "late package"])
    assert docs.dtype == np.float32
    assert np.allclose(np.linalg.norm(docs, axis=1), 1.0, atol=1e-6)
    assert np.allclose(embedder.embed_query("refund policy"), docs[0], atol=1e-6)
    with KBIndex(":memory:", embedder) as index:
        index.ingest(seed_kb())
        results = HybridRetriever(index, mode="vector").search(RetrievalQuery(text="refund"))
        assert results
