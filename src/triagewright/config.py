"""12-factor settings. Every value comes from the environment (or ``.env``) with a safe default.

Secrets are ``SecretStr`` and never logged. The provider defaults to ``replay``
so nothing calls the network unless you opt in. The live providers are the
Gemini API free tier and OpenRouter ``:free`` models, and settings reject any
model id that isn't free on the chosen provider (owner decision, ADR-001).
Blank secrets count as unset.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from triagewright.contracts.agents import LoopConfig
from triagewright.contracts.guardrails import PolicyConfig
from triagewright.contracts.llm import Provider

# Model ids per purpose when ``TW_MODEL_*`` is unset. Replay uses the Gemini
# table, because recordings are made with Gemini; the fingerprint includes the
# model id, so changing a default means re-recording.
DEFAULT_MODELS: dict[Provider, dict[str, str]] = {
    Provider.GEMINI: {
        "coordinator": "gemini-3.5-flash-lite",
        "triage": "gemini-3.5-flash-lite",
        "answer": "gemini-3.5-flash-lite",
        "action": "gemini-3.5-flash-lite",
        "escalation": "gemini-3.5-flash-lite",
        "judge": "gemini-3.8-flash",
    },
    Provider.OPENROUTER: {
        "coordinator": "google/gemma-4-31b-it:free",
        "triage": "google/gemma-4-31b-it:free",
        "answer": "google/gemma-4-31b-it:free",
        "action": "google/gemma-4-31b-it:free",
        "escalation": "google/gemma-4-31b-it:free",
        "judge": "nvidia/nemotron-3-ultra-550b-a55b:free",
    },
}
DEFAULT_MODELS[Provider.REPLAY] = DEFAULT_MODELS[Provider.GEMINI]

# Gemini families with a free tier: numbered Flash and Flash-Lite (and their
# previews), and Gemma. No ``-latest`` aliases (they drift) and no Pro or
# other paid-only models.
_GEMINI_FREE = re.compile(
    r"(?!.*-latest$)(?:gemini-\d+(?:\.\d+)?-flash(?:-lite)?(?:-preview)?|gemma-\d+[a-z0-9.-]*)"
)


def is_free_model(provider: Provider, model: str) -> bool:
    """Whether ``model`` is a free model on ``provider``. Adapters refuse any other id."""
    if provider is Provider.OPENROUTER:
        return model.endswith(":free")
    if provider is Provider.GEMINI:
        return _GEMINI_FREE.fullmatch(model) is not None
    return is_free_model(Provider.GEMINI, model) or is_free_model(Provider.OPENROUTER, model)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TW_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ---- model provider ----
    llm_provider: Provider = Provider.REPLAY
    openrouter_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("OPENROUTER_API_KEY", "TW_OPENROUTER_API_KEY")
    )
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    # The key's Google project must be on the free plan (no billing linked).
    gemini_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("GEMINI_API_KEY", "TW_GEMINI_API_KEY")
    )
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    recordings_dir: Path = Path("evals/fixtures/recordings")
    record: bool = False  # live calls also append to recordings_dir

    # Model id overrides per purpose. Unset means DEFAULT_MODELS[llm_provider].
    model_coordinator: str | None = None
    model_triage: str | None = None
    model_answer: str | None = None
    model_action: str | None = None
    model_escalation: str | None = None
    model_judge: str | None = None

    # ---- agent loop ----
    max_turns: int = 8
    turn_timeout_s: float = 20.0
    total_timeout_s: float = 30.0
    handoff_reserve_s: float = 3.0
    max_tokens: int = 1024
    tool_retries: int = 2
    retry_backoff_s: float = 0.5

    # ---- policy ----
    refund_hard_limit_cents: int = 50_000
    refund_window_days: int = 30
    max_refunds_per_customer_30d: int = 2

    # ---- approvals ----
    # HMAC key shared by the executor and the MCP server, at least 32 chars.
    # Required unless the MCP transport is in-process (wiring then makes an
    # ephemeral key); the MCP server refuses to start over stdio/http without it.
    approval_key: SecretStr | None = None
    # Bearer token the API/CLI require for approval decisions. Unset means
    # decisions are disabled.
    approver_token: SecretStr | None = None

    # ---- data and storage ----
    seed_dir: Path = Path("data/seed")
    kb_dir: Path = Path("data/kb")
    db_path: Path = Path("var/triagewright.sqlite")
    embedder: Literal["hashing", "fastembed"] = "hashing"

    # ---- MCP ----
    mcp_transport: Literal["inprocess", "stdio", "http"] = "inprocess"
    mcp_url: str = "http://localhost:8765/mcp"

    # ---- observability ----
    trace_exporter: Literal["none", "jsonl", "otlp"] = "jsonl"
    trace_dir: Path = Path("var/traces")
    otlp_endpoint: str = "http://localhost:4318"
    pricing_file: Path = Path("config/pricing.toml")

    # ---- API ----
    api_host: str = "127.0.0.1"
    api_port: int = 8080

    @field_validator(
        "openrouter_api_key",
        "gemini_api_key",
        "approval_key",
        "approver_token",
        *(f"model_{purpose}" for purpose in DEFAULT_MODELS[Provider.GEMINI]),
        mode="before",
    )
    @classmethod
    def _blank_is_unset(cls, value: Any) -> Any:
        return None if isinstance(value, str) and not value.strip() else value

    @model_validator(mode="after")
    def _check(self) -> Settings:
        paid = [m for m in self.models().values() if not is_free_model(self.llm_provider, m)]
        if paid:
            raise ValueError(f"only free {self.llm_provider} models are allowed, got {paid}")
        key = self.approval_key
        if key is not None and len(key.get_secret_value()) < 32:
            raise ValueError("TW_APPROVAL_KEY must be at least 32 characters")
        if self.mcp_transport != "inprocess" and key is None:
            raise ValueError("TW_APPROVAL_KEY is required when TW_MCP_TRANSPORT is not inprocess")
        return self

    def loop_config(self) -> LoopConfig:
        return LoopConfig(
            max_turns=self.max_turns,
            turn_timeout_s=self.turn_timeout_s,
            total_timeout_s=self.total_timeout_s,
            handoff_reserve_s=self.handoff_reserve_s,
            max_tokens=self.max_tokens,
            tool_retries=self.tool_retries,
            retry_backoff_s=self.retry_backoff_s,
        )

    def policy_config(self) -> PolicyConfig:
        return PolicyConfig(
            refund_hard_limit_cents=self.refund_hard_limit_cents,
            refund_window_days=self.refund_window_days,
            max_refunds_per_customer_30d=self.max_refunds_per_customer_30d,
        )

    def models(self) -> dict[str, str]:
        """Model id per purpose: the ``TW_MODEL_*`` override, else the provider default."""
        defaults = DEFAULT_MODELS[self.llm_provider]
        return {p: getattr(self, f"model_{p}") or m for p, m in defaults.items()}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
