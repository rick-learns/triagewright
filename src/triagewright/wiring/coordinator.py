"""The coordinator over already-built lanes, configured from ``Settings``."""

from __future__ import annotations

from typing import Literal

from triagewright.agent.coordinator import PURPOSES, Coordinator
from triagewright.config import Settings
from triagewright.contracts.llm import Provider
from triagewright.contracts.ports import (
    ActionExecutor,
    Guardrails,
    ModelClient,
    TelemetryFactory,
    ToolGateway,
)


def _mode(settings: Settings) -> Literal["live", "replay"]:
    return "replay" if settings.llm_provider is Provider.REPLAY else "live"


def _coordinator(
    settings: Settings,
    *,
    model: ModelClient,
    gateway: ToolGateway,
    guardrails: Guardrails,
    executor: ActionExecutor,
    telemetry: TelemetryFactory,
) -> Coordinator:
    models = settings.models()
    return Coordinator(
        model=model,
        gateway=gateway,
        guardrails=guardrails,
        executor=executor,
        telemetry=telemetry,
        models={p: models[p] for p in PURPOSES},
        config=settings.loop_config(),
        mode=_mode(settings),
    )
