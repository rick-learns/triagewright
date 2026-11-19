from __future__ import annotations

from collections.abc import Iterator

import pytest

from triagewright.rag import HashingEmbedder, KBIndex
from triagewright.testing import seed_kb


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def kb_index() -> Iterator[KBIndex]:
    with KBIndex(":memory:", HashingEmbedder()) as index:
        index.ingest(seed_kb())
        yield index
