"""Text embedders: a deterministic hashing embedder (default) and optional ``fastembed``.

Vectors are float32 and L2-normalized, so a dot product is the cosine. An
all-zero vector (no usable terms) scores zero against everything.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from triagewright.rag.text import terms

type Vectors = NDArray[np.float32]


class Embedder(Protocol):
    """``name`` identifies the vector space; an index built with one name is unusable by another."""

    @property
    def name(self) -> str: ...

    def embed_documents(self, texts: Sequence[str]) -> Vectors: ...

    def embed_query(self, text: str) -> Vectors: ...


def _normalize(rows: NDArray[np.float64]) -> Vectors:
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    unit: NDArray[np.float64] = np.divide(rows, norms, out=np.zeros_like(rows), where=norms > 0)
    return unit.astype(np.float32)


class HashingEmbedder:
    """Signed feature hashing of words and character trigrams. No model, no download.

    Uses blake2b rather than ``hash()``, so vectors are identical across runs
    and machines. Trigrams let ``refund`` match ``refunds`` and ``refunded``.
    """

    def __init__(self, dim: int = 512) -> None:
        if dim < 16:
            raise ValueError("dim must be at least 16")
        self.dim = dim

    @property
    def name(self) -> str:
        return f"hashing-v1-{self.dim}"

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        rows = np.zeros((len(texts), self.dim), dtype=np.float64)
        for row, text in enumerate(texts):
            for feature, weight in _features(text):
                h = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest())
                rows[row, h % self.dim] += -weight if h >> 63 else weight
        return _normalize(rows)

    def embed_query(self, text: str) -> Vectors:
        row: Vectors = self.embed_documents([text])[0]
        return row


def _features(text: str) -> Iterable[tuple[str, float]]:
    for term in terms(text):
        yield f"w:{term}", 1.0
        padded = f"<{term}>"
        for i in range(len(padded) - 2):
            yield f"c:{padded[i : i + 3]}", 0.3


class FastEmbedEmbedder:
    """Local ONNX embeddings via ``fastembed`` (the ``local-embeddings`` extra).

    Imported lazily, so the package works without it. The first use downloads
    the model to fastembed's cache.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise RuntimeError(
                "fastembed is not installed; run `uv sync --extra local-embeddings` "
                "or set TW_EMBEDDER=hashing"
            ) from exc
        self.model_name = model_name
        self._model: Any = TextEmbedding(model_name=model_name)

    @property
    def name(self) -> str:
        return f"fastembed:{self.model_name}"

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return _normalize(np.array(list(self._model.passage_embed(list(texts))), dtype=np.float64))

    def embed_query(self, text: str) -> Vectors:
        rows = np.array(list(self._model.query_embed([text])), dtype=np.float64)
        row: Vectors = _normalize(rows)[0]
        return row


def make_embedder(kind: Literal["hashing", "fastembed"]) -> Embedder:
    """The embedder named by ``Settings.embedder``."""
    return FastEmbedEmbedder() if kind == "fastembed" else HashingEmbedder()
