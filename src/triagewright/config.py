"""12-factor settings. Every value comes from the environment (or ``.env``) with a safe default.

Secrets are ``SecretStr`` and never logged. The provider defaults to ``replay``
so nothing calls a paid API unless you opt in.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
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
    anthropic_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("ANTHROPIC_API_KEY", "TW_ANTHROPIC_API_KEY")
    )
    anthropic_base_url: str | None = None
    openrouter_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("OPENROUTER_API_KEY", "TW_OPENROUTER_API_KEY")
    )
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    recordings_dir: Path = Path("evals/fixtures/recordings")
    record: bool = False  # live calls also append to recordings_dir

    # Model ids per purpose. Defaults are Anthropic ids; override per provider in .env.
    model_coordinator: str = "claude-haiku-5-5"
    model_triage: str = "claude-haiku-5-5"
    model_answer: str = "claude-haiku-5-5"
    model_action: str = "claude-haiku-5-5"
    model_escalation: str = "claude-haiku-5-5"
    model_judge: str = "claude-sonnet-5-5"

    # ---- agent loop ----
    max_turns: int = 8
    turn_timeout_s: float = 20.0
    total_timeout_s: float = 30.0
    max_tokens: int = 1024

    # ---- policy ----
    refund_hard_limit_cents: int = 50_000
    refund_window_days: int = 30
    max_refunds_per_customer_30d: int = 2

    # ---- data and storage ----
    seed_dir: Path = Path("data/seed")
    kb_dir: Path = Path("data/kb")
    db_path: Path = Path("var/triagewright.sqlite")
    embedder: Literal["hashing", "fastembed"] = "hashing"

    # ---- MCP ----
    mcp_transport: Literal["inprocess", "stdio", "http"] = "inprocess"
    mcp_url: str = "http://localhost:8765/mcp"

    # ---- observability ----
    trace_exporter: Literal["none", "console", "jsonl", "otlp"] = "jsonl"
    trace_dir: Path = Path("var/traces")
    otlp_endpoint: str = "http://localhost:4318"
    pricing_file: Path = Path("config/pricing.toml")

    # ---- API ----
    api_host: str = "127.0.0.1"
    api_port: int = 8080

    def loop_config(self) -> LoopConfig:
        return LoopConfig(
            max_turns=self.max_turns,
            turn_timeout_s=self.turn_timeout_s,
            total_timeout_s=self.total_timeout_s,
            max_tokens=self.max_tokens,
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
