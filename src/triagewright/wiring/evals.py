"""The eval stack: a fresh in-memory world, approval store, and coordinator per case."""

from __future__ import annotations

import secrets
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from triagewright.config import Settings
from triagewright.contracts.evals import ToolFault
from triagewright.contracts.guardrails import ApprovalRequest
from triagewright.contracts.ports import Retriever, TelemetryFactory, TicketResolver
from triagewright.contracts.tools import ToolCallRecord
from triagewright.contracts.world import WorldState
from triagewright.data import load_kb
from triagewright.guardrails import GuardedActionExecutor, RuleGuardrails, SqliteApprovalStore
from triagewright.mcp_server import InProcessToolGateway
from triagewright.observability import OTelTelemetryFactory
from triagewright.rag import HybridRetriever, KBIndex, make_embedder
from triagewright.wiring.coordinator import _coordinator
from triagewright.wiring.models import ModelFactory, _model_factory


@dataclass(frozen=True)
class CaseStack:
    """One eval case's real stack. Satisfies ``ports.CaseRuntime``."""

    resolver: TicketResolver
    gateway: InProcessToolGateway
    approvals: SqliteApprovalStore

    def tool_calls(self) -> list[ToolCallRecord]:
        return list(self.gateway.calls)

    def ledger(self) -> WorldState:
        return self.gateway.ledger()

    def pending_approvals(self) -> list[ApprovalRequest]:
        return self.approvals.list_pending()


def memory_retriever(settings: Settings) -> HybridRetriever:
    """The KB at ``settings.kb_dir``, indexed in memory (evals never touch ``var/``)."""
    if not settings.kb_dir.is_dir():
        raise FileNotFoundError(f"KB directory not found: {settings.kb_dir}")
    index = KBIndex(":memory:", make_embedder(settings.embedder))
    index.ingest(load_kb(settings.kb_dir))
    return HybridRetriever(index)


class EvalStackFactory:
    """Builds a fresh :class:`CaseStack` per case. Satisfies ``ports.CaseRuntimeFactory``."""

    def __init__(
        self,
        settings: Settings,
        *,
        model_factory: ModelFactory | None = None,
        telemetry: TelemetryFactory | None = None,
        retriever: Retriever | None = None,
    ) -> None:
        self.settings = settings
        self._models = model_factory or _model_factory(settings)
        self._telemetry = telemetry or OTelTelemetryFactory.from_settings(settings)
        self._retriever = retriever or memory_retriever(settings)
        self._guardrails = RuleGuardrails()
        self._key = secrets.token_bytes(32)  # signs and checks within each case's stack

    def build(
        self, *, case_id: str, world: WorldState, as_of: datetime, faults: Sequence[ToolFault]
    ) -> CaseStack:
        gateway = InProcessToolGateway.for_world(
            world,
            as_of=as_of,
            approval_key=self._key,
            retriever=self._retriever,
            faults=faults,
            policy=self.settings.policy_config(),
        )
        approvals = SqliteApprovalStore()  # in memory
        executor = GuardedActionExecutor(
            gateway,
            self._guardrails,
            approvals,
            approval_key=self._key,
            clock=lambda: as_of,
            loop=self.settings.loop_config(),
        )
        resolver = _coordinator(
            self.settings,
            model=self._models(case_id),
            gateway=gateway,
            guardrails=self._guardrails,
            executor=executor,
            telemetry=self._telemetry,
        )
        return CaseStack(resolver=resolver, gateway=gateway, approvals=approvals)
