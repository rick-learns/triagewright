from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from tests.observability.support import PAID, prices, response
from triagewright.config import DEFAULT_MODELS, Settings
from triagewright.contracts.llm import Provider, Usage
from triagewright.observability import OTelTelemetryFactory, PriceTable, UnpricedModelError

SHIPPED = Path("config/pricing.toml")


def test_every_default_model_is_priced_at_zero() -> None:
    table = PriceTable.load(SHIPPED)
    big = Usage(
        input_tokens=10**9,
        output_tokens=10**9,
        cache_read_input_tokens=10**9,
        cache_creation_input_tokens=10**9,
    )
    for provider, models in DEFAULT_MODELS.items():
        for model in models.values():
            price = table.get(provider, model)
            assert price is not None, (provider, model)
            assert price.source.startswith("https://")
            rates = (
                price.input_per_mtok_usd,
                price.output_per_mtok_usd,
                price.cache_read_per_mtok_usd,
                price.cache_write_per_mtok_usd,
            )
            assert rates == (0, 0, 0, 0), (provider, model)
            assert table.cost(provider, model, big) == 0.0


def test_cost_uses_usage_buckets_per_million() -> None:
    usage = Usage(
        input_tokens=2_000_000,  # excludes cache tokens (Anthropic semantics)
        output_tokens=500_000,
        cache_read_input_tokens=400_000,
        cache_creation_input_tokens=100_000,
    )
    # 2 * 1.0 + 0.5 * 4.0 + 0.4 * 0.25 + 0.1 * 2.0
    assert prices().cost(Provider.GEMINI, PAID, usage) == pytest.approx(4.3)


def test_unknown_model_has_no_price() -> None:
    table = prices()
    assert table.get(Provider.GEMINI, "gemini-9-flash") is None
    assert table.cost(Provider.GEMINI, "gemini-9-flash", Usage(input_tokens=1)) is None
    assert table.response_cost(response(model="gemini-9-flash")) is None
    # Same id, wrong provider: still unknown.
    assert table.get(Provider.OPENROUTER, PAID) is None


def test_provider_reported_cost_wins() -> None:
    table = prices()
    assert table.response_cost(response(provider_cost_usd=0.0)) == 0.0
    assert table.response_cost(response(model="unlisted", provider_cost_usd=0.25)) == 0.25


def test_replay_uses_the_recording_provider_price() -> None:
    table = prices()
    assert table.get(Provider.REPLAY, PAID) == table.get(Provider.GEMINI, PAID)
    assert table.get(Provider.REPLAY, "unlisted") is None


def test_check_rejects_unpriced_configured_models() -> None:
    table = prices()
    table.check(Provider.GEMINI, {"answer": PAID})
    with pytest.raises(UnpricedModelError, match="gemini-9-flash"):
        table.check(Provider.GEMINI, {"answer": PAID, "judge": "gemini-9-flash"})


def test_factory_from_settings_fails_fast_on_unpriced_override() -> None:
    settings = Settings(_env_file=None, model_answer="gemini-9-flash", trace_exporter="none")
    with pytest.raises(UnpricedModelError):
        OTelTelemetryFactory.from_settings(settings)


def test_load_rejects_duplicates_and_missing_source(tmp_path: Path) -> None:
    row = (
        '[[model]]\nprovider = "gemini"\nmodel = "m"\n'
        "input_per_mtok_usd = 0.0\noutput_per_mtok_usd = 0.0\n"
    )
    missing = tmp_path / "missing.toml"
    missing.write_text(row)
    with pytest.raises(ValidationError, match="source"):
        PriceTable.load(missing)
    dup = tmp_path / "dup.toml"
    dup.write_text((row + 'source = "x"\n') * 2)
    with pytest.raises(ValueError, match="duplicate"):
        PriceTable.load(dup)
    empty = tmp_path / "empty.toml"
    empty.write_text("")
    assert PriceTable.load(empty).get(Provider.GEMINI, "m") is None
