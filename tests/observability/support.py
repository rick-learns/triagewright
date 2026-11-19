"""Shared builders for the observability tests."""

from __future__ import annotations

from triagewright.contracts.llm import (
    Message,
    ModelRequest,
    ModelResponse,
    Provider,
    StopReason,
    TextBlock,
    Usage,
)
from triagewright.contracts.telemetry import ModelPrice
from triagewright.observability import PriceTable

PAID = "paid-model"  # a test-only priced model, so cost math is non-trivial


class FakeClock:
    """Monotonic seconds that only move when told to."""

    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t

    def advance(self, ms: float) -> None:
        self.t += ms / 1000


def prices() -> PriceTable:
    return PriceTable(
        [
            ModelPrice(
                model=PAID,
                provider=Provider.GEMINI,
                input_per_mtok_usd=1.0,
                output_per_mtok_usd=4.0,
                cache_read_per_mtok_usd=0.25,
                cache_write_per_mtok_usd=2.0,
                source="test",
            ),
            ModelPrice(
                model="free-model",
                provider=Provider.GEMINI,
                input_per_mtok_usd=0.0,
                output_per_mtok_usd=0.0,
                source="test",
            ),
        ]
    )


def request(purpose: str = "coordinator", model: str = PAID) -> ModelRequest:
    return ModelRequest(
        model=model,
        system="s",
        messages=[Message(role="user", content=[TextBlock(text="hi")])],
        purpose=purpose,
    )


def response(
    *,
    model: str = PAID,
    provider: Provider = Provider.GEMINI,
    latency_ms: float = 100.0,
    ttft_ms: float | None = None,
    usage: Usage | None = None,
    provider_cost_usd: float | None = None,
    stop_reason: StopReason = StopReason.END_TURN,
) -> ModelResponse:
    return ModelResponse(
        id="msg_1",
        provider=provider,
        model=model,
        content=[TextBlock(text="ok")],
        stop_reason=stop_reason,
        usage=usage or Usage(input_tokens=1000, output_tokens=200),
        latency_ms=latency_ms,
        ttft_ms=ttft_ms,
        provider_cost_usd=provider_cost_usd,
    )
