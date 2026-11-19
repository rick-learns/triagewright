"""Provider-neutral model I/O, shaped like the Anthropic Messages API.

Adapters (Anthropic, OpenRouter, replay) translate to and from these types, so
the agent loop handles ``stop_reason`` and ``tool_use`` blocks the same way on
every provider. Recordings let CI replay model calls without API keys.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field

from triagewright.contracts.common import Contract, JsonObject


class Provider(StrEnum):
    OPENROUTER = "openrouter"
    REPLAY = "replay"


class StopReason(StrEnum):
    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    PAUSE_TURN = "pause_turn"
    REFUSAL = "refusal"


class TextBlock(Contract):
    type: Literal["text"] = "text"
    text: str


class ToolUseBlock(Contract):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: JsonObject


class ToolResultBlock(Contract):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str  # JSON of a ToolResult (or a subagent output) as text
    is_error: bool = False


type ContentBlock = Annotated[
    TextBlock | ToolUseBlock | ToolResultBlock, Field(discriminator="type")
]


class Message(Contract):
    role: Literal["user", "assistant"]
    content: list[ContentBlock]


class ToolDefinition(Contract):
    name: Annotated[str, Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")]
    description: str
    input_schema: JsonObject


class ToolChoice(Contract):
    type: Literal["auto", "any", "tool", "none"] = "auto"
    name: str | None = None


class ModelRequest(Contract):
    model: str
    system: str
    messages: Annotated[list[Message], Field(min_length=1)]
    tools: list[ToolDefinition] = []
    tool_choice: ToolChoice | None = None
    max_tokens: Annotated[int, Field(ge=1, le=64_000)] = 1024
    temperature: Annotated[float, Field(ge=0.0, le=1.0)] | None = 0.0
    purpose: str = "unspecified"  # e.g. "coordinator", "triage"; not sent to providers

    def fingerprint(self) -> str:
        """Stable hash of everything that affects the model output (excludes ``purpose``)."""
        payload = self.model_dump(mode="json", exclude={"purpose"})
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


class Usage(Contract):
    """Anthropic semantics: ``input_tokens`` excludes cache reads and writes.

    Adapters for providers that fold cache tokens into prompt tokens subtract them.
    """

    input_tokens: Annotated[int, Field(ge=0)] = 0
    output_tokens: Annotated[int, Field(ge=0)] = 0
    cache_read_input_tokens: Annotated[int, Field(ge=0)] = 0
    cache_creation_input_tokens: Annotated[int, Field(ge=0)] = 0


class ModelResponse(Contract):
    id: str
    provider: Provider
    model: str
    content: list[ContentBlock]
    stop_reason: StopReason
    usage: Usage
    latency_ms: Annotated[float, Field(ge=0)]
    ttft_ms: Annotated[float, Field(ge=0)] | None = None
    provider_cost_usd: float | None = None  # only when the provider reports it


class Recording(Contract):
    """One recorded model call, stored as JSONL in ``evals/fixtures/recordings/<case_id>.jsonl``.

    Replay looks up by ``fingerprint`` within the case file. Identical
    fingerprints are served in ``sequence`` order. A miss is an error, never a
    live call. The fingerprint covers the model id, so switching models means
    re-recording. Determinism rests on ``common.canonical_json`` for all
    model-visible content and ``common.stable_id`` for all model-visible ids.
    """

    case_id: str | None = None
    sequence: Annotated[int, Field(ge=0)] = 0
    fingerprint: str
    request: ModelRequest
    response: ModelResponse
