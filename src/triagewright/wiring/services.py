"""The API and CLI bundle: the real lanes over a persistent world and approval store."""

from __future__ import annotations

import secrets

from triagewright.api.services import Services, WiringUnavailableError
from triagewright.config import Settings
from triagewright.contracts.llm import Provider
from triagewright.contracts.ports import TelemetryFactory, ToolGateway
from triagewright.data import load_as_of
from triagewright.guardrails import GuardedActionExecutor, RuleGuardrails, SqliteApprovalStore
from triagewright.mcp_server import (
    InProcessToolGateway,
    McpToolGateway,
    ToolService,
    build_service,
)
from triagewright.observability import OTelTelemetryFactory
from triagewright.wiring.coordinator import _coordinator
from triagewright.wiring.gateway import BackgroundMcpGateway
from triagewright.wiring.models import ModelFactory, _model_factory, _SharedHttp


def _approval_key(settings: Settings) -> bytes:
    """``TW_APPROVAL_KEY``, or an ephemeral key when only this process signs and checks."""
    if settings.approval_key is not None:
        return settings.approval_key.get_secret_value().encode()
    if settings.mcp_transport != "inprocess":  # Settings already refuses this
        raise WiringUnavailableError("TW_APPROVAL_KEY is required for an out-of-process MCP server")
    return secrets.token_bytes(32)


def build_services(
    settings: Settings,
    *,
    model_factory: ModelFactory | None = None,
    telemetry: TelemetryFactory | None = None,
) -> Services:
    """The real lanes for the API and CLI. ``model_factory`` and ``telemetry`` are for tests."""
    if model_factory is None and settings.llm_provider is Provider.REPLAY:
        raise WiringUnavailableError(
            "TW_LLM_PROVIDER=replay only replays eval-case recordings; set TW_LLM_PROVIDER "
            "to gemini or openrouter (free models) to resolve tickets"
        )
    key = _approval_key(settings)
    http = _SharedHttp(settings)
    service: ToolService | None = None
    gateway: ToolGateway
    if settings.mcp_transport == "inprocess":
        service = build_service(settings, approval_key=key)
        gateway = InProcessToolGateway(service, log_calls=False)
    else:
        gateway = BackgroundMcpGateway(
            lambda: McpToolGateway.from_settings(settings, log_calls=False)
        )
    as_of = load_as_of(settings.seed_dir)
    guardrails = RuleGuardrails()
    approvals = SqliteApprovalStore(settings.approvals_db_path)
    executor = GuardedActionExecutor(
        gateway,
        guardrails,
        approvals,
        approval_key=key,
        clock=lambda: as_of,
        loop=settings.loop_config(),
    )
    tracing: OTelTelemetryFactory | None = None
    if telemetry is None:
        telemetry = tracing = OTelTelemetryFactory.from_settings(settings)
    resolver = _coordinator(
        settings,
        model=(model_factory or _model_factory(settings, http))(None),
        gateway=gateway,
        guardrails=guardrails,
        executor=executor,
        telemetry=telemetry,
    )

    def close() -> None:
        if isinstance(gateway, BackgroundMcpGateway):
            gateway.close()
        if service is not None:
            service.close()
        approvals.close()
        if tracing is not None:
            tracing.shutdown()  # flush spans

    return Services(
        resolver=resolver,
        executor=executor,
        approvals=approvals,
        aclose=http.aclose,
        close=close,
    )
