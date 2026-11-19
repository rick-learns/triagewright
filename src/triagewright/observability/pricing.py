"""Model prices from ``config/pricing.toml`` and per-call cost.

An unknown model has no price (``None``), never a guessed one. A cost the
provider reports on the response wins over the table.
"""

from __future__ import annotations

import tomllib
from collections.abc import Iterable, Mapping
from pathlib import Path

from triagewright.contracts.llm import ModelResponse, Provider, Usage
from triagewright.contracts.telemetry import ModelPrice

# Replay serves recordings made on a live provider, so it uses that provider's prices.
_REPLAY_SOURCES = (Provider.GEMINI, Provider.OPENROUTER)


class UnpricedModelError(LookupError):
    """A model call has no provider-reported cost and no entry in the pricing table."""


class PriceTable:
    def __init__(self, prices: Iterable[ModelPrice]) -> None:
        self._prices: dict[tuple[Provider, str], ModelPrice] = {}
        for price in prices:
            key = (price.provider, price.model)
            if key in self._prices:
                raise ValueError(f"duplicate price for {price.provider}/{price.model}")
            self._prices[key] = price

    @classmethod
    def load(cls, path: Path) -> PriceTable:
        with path.open("rb") as f:
            rows = tomllib.load(f).get("model", [])
        return cls(ModelPrice.model_validate(row) for row in rows)

    def get(self, provider: Provider, model: str) -> ModelPrice | None:
        if (price := self._prices.get((provider, model))) is not None:
            return price
        if provider is Provider.REPLAY:
            for source in _REPLAY_SOURCES:
                if (price := self._prices.get((source, model))) is not None:
                    return price
        return None

    def cost(self, provider: Provider, model: str, usage: Usage) -> float | None:
        """USD for ``usage`` on this model, or ``None`` if the model is unknown."""
        price = self.get(provider, model)
        if price is None:
            return None
        return (
            usage.input_tokens * price.input_per_mtok_usd
            + usage.output_tokens * price.output_per_mtok_usd
            + usage.cache_read_input_tokens * price.cache_read_per_mtok_usd
            + usage.cache_creation_input_tokens * price.cache_write_per_mtok_usd
        ) / 1_000_000

    def response_cost(self, response: ModelResponse) -> float | None:
        """``provider_cost_usd`` when the provider reported it, else the table price."""
        if response.provider_cost_usd is not None:
            return response.provider_cost_usd
        return self.cost(response.provider, response.model, response.usage)

    def unpriced(self, provider: Provider, models: Iterable[str]) -> list[str]:
        """The models in ``models`` that have no price on ``provider``."""
        return sorted({m for m in models if self.get(provider, m) is None})

    def check(self, provider: Provider, models: Mapping[str, str]) -> None:
        """Raise :class:`UnpricedModelError` if any configured model (purpose -> id) is unknown."""
        if missing := self.unpriced(provider, models.values()):
            raise UnpricedModelError(f"no price in the pricing table for {provider}: {missing}")
