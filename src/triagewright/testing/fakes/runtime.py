"""Per-case fake stack (``CaseRuntime``) and the factory that builds one per eval case."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime

from triagewright.contracts.evals import ToolFault
from triagewright.contracts.guardrails import (
    ApprovalRequest,
    PolicyConfig,
)
from triagewright.contracts.ports import TicketResolver
from triagewright.contracts.rag import KBDocument
from triagewright.contracts.tools import (
    ToolCallRecord,
)
from triagewright.contracts.world import (
    WorldState,
)
from triagewright.testing.fakes.approvals import InMemoryApprovalStore
from triagewright.testing.fakes.executor import FakeActionExecutor
from triagewright.testing.fakes.gateway import FakeToolGateway
from triagewright.testing.fakes.guardrails import PassthroughGuardrails
from triagewright.testing.fakes.resolver import StubResolver
from triagewright.testing.fakes.retrieval import FakeRetriever


@dataclass
class FakeCaseRuntime:
    """One isolated fake stack. Satisfies the ``CaseRuntime`` port."""

    gateway: FakeToolGateway
    approvals: InMemoryApprovalStore
    executor: FakeActionExecutor
    resolver: TicketResolver = field(default_factory=lambda: StubResolver())

    def tool_calls(self) -> list[ToolCallRecord]:
        return list(self.gateway.calls)

    def ledger(self) -> WorldState:
        return self.gateway.ledger()

    def pending_approvals(self) -> list[ApprovalRequest]:
        return self.approvals.list_pending()


class FakeCaseRuntimeFactory:
    """Builds a fresh fake stack per case. ``make_resolver`` wires a resolver onto it."""

    def __init__(
        self,
        make_resolver: Callable[[FakeCaseRuntime], TicketResolver] | None = None,
        *,
        docs: Sequence[KBDocument] = (),
        policy: PolicyConfig | None = None,
    ) -> None:
        self.built: list[str] = []  # case ids, in build order
        self._make_resolver = make_resolver
        self._retriever = FakeRetriever(docs) if docs else None
        self._policy = policy

    def build(
        self, *, case_id: str, world: WorldState, as_of: datetime, faults: Sequence[ToolFault]
    ) -> FakeCaseRuntime:
        self.built.append(case_id)
        gateway = FakeToolGateway(
            world, as_of=as_of, retriever=self._retriever, faults=faults, policy=self._policy
        )
        approvals = InMemoryApprovalStore()
        executor = FakeActionExecutor(gateway, PassthroughGuardrails(), approvals, now=as_of)
        runtime = FakeCaseRuntime(gateway=gateway, approvals=approvals, executor=executor)
        if self._make_resolver is not None:
            runtime.resolver = self._make_resolver(runtime)
        return runtime
