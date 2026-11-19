"""Scripted model client that replays pre-built responses."""

from __future__ import annotations

from collections.abc import Sequence

from triagewright.contracts.llm import ModelRequest, ModelResponse


class ScriptedModelClient:
    """Returns pre-built responses in order and records every request."""

    def __init__(self, responses: Sequence[ModelResponse]) -> None:
        self._responses = list(responses)
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if not self._responses:
            raise RuntimeError("ScriptedModelClient ran out of responses")
        return self._responses.pop(0)
