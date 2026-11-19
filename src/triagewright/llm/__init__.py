"""Lane B: model clients (Gemini, OpenRouter, replay/record) behind ``ports.ModelClient``."""

from triagewright.llm.errors import (
    ModelAuthError,
    ModelConfigError,
    ModelError,
    ModelRequestError,
    ModelResponseError,
    ModelUnavailableError,
    ReplayMissError,
)
from triagewright.llm.factory import ModelClients, build_model_clients
from triagewright.llm.openai_compat import GeminiClient, OpenAICompatClient, OpenRouterClient
from triagewright.llm.replay import RecordingClient, RecordingFile, ReplayClient, recording_path

__all__ = [
    "GeminiClient",
    "ModelAuthError",
    "ModelClients",
    "ModelConfigError",
    "ModelError",
    "ModelRequestError",
    "ModelResponseError",
    "ModelUnavailableError",
    "OpenAICompatClient",
    "OpenRouterClient",
    "RecordingClient",
    "RecordingFile",
    "ReplayClient",
    "ReplayMissError",
    "build_model_clients",
    "recording_path",
]
