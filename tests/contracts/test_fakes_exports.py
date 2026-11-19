"""The fakes package keeps the public surface it had as a single module."""

from __future__ import annotations

import triagewright.testing as testing
import triagewright.testing.fakes as fakes

FAKES = {
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
}
FIXTURES = {
    "REPO_ROOT",
    "model_response",
    "sample_ticket",
    "seed_as_of",
    "seed_kb",
    "seed_world",
    "text_block",
    "tool_use_block",
}


def test_fakes_all_is_exact_and_matches_testing() -> None:
    assert set(fakes.__all__) == FAKES
    assert set(testing.__all__) == FAKES | FIXTURES
    for name in FAKES:
        assert getattr(testing, name) is getattr(fakes, name)
