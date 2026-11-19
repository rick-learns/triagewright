"""Settings and scripted models shared by the wiring tests."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from triagewright import wiring
from triagewright.config import Settings
from triagewright.contracts.llm import ModelRequest, ModelResponse
from triagewright.testing import ScriptedModelClient, sample_ticket

REFUND_TICKET = sample_ticket("Please refund $50.00 for order ord_a1001, it arrived broken.")


def settings(tmp_path: Path, **overrides: object) -> Settings:
    return Settings(
        _env_file=None,
        db_path=tmp_path / "world.sqlite",
        approvals_db_path=tmp_path / "approvals.sqlite",
        trace_exporter="none",
        **overrides,  # type: ignore[arg-type]
    )


class Scripted:
    """Scripted responses, stamped with the requested model so the real price table applies."""

    def __init__(self, responses: Sequence[ModelResponse]) -> None:
        self.inner = ScriptedModelClient(responses)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        response = await self.inner.complete(request)
        return response.model_copy(update={"model": request.model})


def scripted(responses: Sequence[ModelResponse]) -> wiring.ModelFactory:
    return lambda case_id: Scripted(responses)
