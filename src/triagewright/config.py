"""12-factor settings. Every value comes from the environment (or ``.env``) with a safe default.

Secrets are ``SecretStr`` and never logged. The provider defaults to ``replay``
so nothing calls the network unless you opt in. The only live provider is
OpenRouter, and every model id must be a ``:free`` model (owner decision,
ADR-001). Blank secrets count as unset.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from triagewright.contracts.agents import LoopConfig
from triagewright.contracts.guardrails import PolicyConfig
from triagewright.contracts.llm import Provider


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
    recordings_dir: Path = Path("evals/fixtures/recordings")
    record: bool = False  # live calls also append to recordings_dir

    # OpenRouter model ids per purpose. Recordings are keyed by model id, so
    # changing one means re-recording. Both support tool calling.
    model_coordinator: str = "google/gemma-4-31b-it:free"
    model_triage: str = "google/gemma-4-31b-it:free"
    model_answer: str = "google/gemma-4-31b-it:free"
    model_action: str = "google/gemma-4-31b-it:free"
    model_escalation: str = "google/gemma-4-31b-it:free"
    model_judge: str = "nvidia/nemotron-3-ultra-550b-a55b:free"

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

    @field_validator("openrouter_api_key", "approval_key", "approver_token", mode="before")
    @classmethod
    def _blank_is_unset(cls, value: Any) -> Any:
        return None if isinstance(value, str) and not value.strip() else value

    @model_validator(mode="after")
    def _check(self) -> Settings:
        paid = [m for m in self.models().values() if not m.endswith(":free")]
        if paid:
            raise ValueError(f"only OpenRouter :free models are allowed, got {paid}")
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
        return {
            "coordinator": self.model_coordinator,
            "triage": self.model_triage,
            "answer": self.model_answer,
            "action": self.model_action,
            "escalation": self.model_escalation,
            "judge": self.model_judge,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
