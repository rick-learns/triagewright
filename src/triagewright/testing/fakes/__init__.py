"""In-memory fakes for every port. Orchestrator-owned; lanes test against these.

They are deliberately simple and deterministic. They are not the real
implementations: the MCP server (lane A), retriever (lane C), guardrails
(lane D), and telemetry (lane E) replace them at integration.
"""

from triagewright.testing.fakes.approvals import InMemoryApprovalStore
from triagewright.testing.fakes.executor import FakeActionExecutor
from triagewright.testing.fakes.gateway import TEST_APPROVAL_KEY, FakeToolGateway
from triagewright.testing.fakes.guardrails import PassthroughGuardrails
from triagewright.testing.fakes.model import ScriptedModelClient
from triagewright.testing.fakes.resolver import StubResolver
from triagewright.testing.fakes.retrieval import FakeRetriever, chunk_documents
from triagewright.testing.fakes.runtime import FakeCaseRuntime, FakeCaseRuntimeFactory
from triagewright.testing.fakes.telemetry import NullTelemetry, NullTelemetryFactory

__all__ = [
    "TEST_APPROVAL_KEY",
    "FakeActionExecutor",
    "FakeCaseRuntime",
    "FakeCaseRuntimeFactory",
    "FakeRetriever",
    "FakeToolGateway",
    "InMemoryApprovalStore",
    "NullTelemetry",
    "NullTelemetryFactory",
    "PassthroughGuardrails",
    "ScriptedModelClient",
    "StubResolver",
    "chunk_documents",
]
