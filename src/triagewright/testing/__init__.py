"""Test doubles and fixtures shared by every lane. Orchestrator-owned."""

from triagewright.testing.fakes import (
    FakeRetriever,
    FakeToolGateway,
    InMemoryApprovalStore,
    NullTelemetry,
    NullTelemetryFactory,
    PassthroughGuardrails,
    ScriptedModelClient,
    StubResolver,
    chunk_documents,
)
from triagewright.testing.fixtures import (
    REPO_ROOT,
    model_response,
    sample_ticket,
    seed_as_of,
    seed_kb,
    seed_world,
    text_block,
    tool_use_block,
)

__all__ = [
    "REPO_ROOT",
    "FakeRetriever",
    "FakeToolGateway",
    "InMemoryApprovalStore",
    "NullTelemetry",
    "NullTelemetryFactory",
    "PassthroughGuardrails",
    "ScriptedModelClient",
    "StubResolver",
    "chunk_documents",
    "model_response",
    "sample_ticket",
    "seed_as_of",
    "seed_kb",
    "seed_world",
    "text_block",
    "tool_use_block",
]
