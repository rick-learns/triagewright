"""The coordinator's ``finalize`` tool: the part of the response a model may supply."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from triagewright.contracts.common import ChunkId, Contract


class CitationCandidate(Contract):
    chunk_id: ChunkId
    quote: Annotated[str, Field(min_length=1, max_length=400)]


class FinalizeArgs(Contract):
    """What the coordinator model may supply. Everything else in the response is code's."""

    status: Literal["resolved", "needs_info", "refused"]
    reply: Annotated[str, Field(min_length=1, max_length=4000)]
    citations: list[CitationCandidate] = []
