"""Builds the model client for each role from ``Settings``. ``triagewright.wiring`` calls this.

- ``replay`` (the default): every role reads the case's recordings; a miss is
  a :class:`ReplayMissError`. Needs ``case_id``.
- ``gemini`` / ``openrouter``: live clients for each role's free model.
- ``TW_RECORD=true`` with a live provider: live clients wrapped to record (and
  resume) into ``recordings_dir``. Needs ``case_id``.

The judge role always uses ``<case_id>.judge.jsonl``; every other role shares
``<case_id>.jsonl``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import httpx

from triagewright.config import Settings
from triagewright.contracts.llm import Provider
from triagewright.contracts.ports import ModelClient
from triagewright.llm.errors import ModelConfigError
from triagewright.llm.openai_compat import GeminiClient, OpenAICompatClient, OpenRouterClient
from triagewright.llm.replay import RecordingClient, RecordingFile, ReplayClient, recording_path

JUDGE_ROLE = "judge"


@dataclass(frozen=True)
class ModelClients:
    """One client per role (``Settings.models()`` keys). Close it to release the HTTP pool."""

    clients: Mapping[str, ModelClient]
    models: Mapping[str, str]
    _http: httpx.AsyncClient | None = None

    def for_role(self, role: str) -> ModelClient:
        return self.clients[role]

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()


def build_model_clients(
    settings: Settings,
    *,
    case_id: str | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> ModelClients:
    """Clients for every role. A passed ``http_client`` is the caller's to close."""
    provider = settings.llm_provider
    models = settings.models()
    files: dict[bool, RecordingFile] = {}

    def recordings(role: str) -> RecordingFile:
        if case_id is None:
            raise ModelConfigError(f"{provider} mode needs a case id to find recordings")
        judge = role == JUDGE_ROLE
        if judge not in files:
            path = recording_path(settings.recordings_dir, case_id, judge=judge)
            files[judge] = RecordingFile(path, case_id=case_id)
        return files[judge]

    if provider is Provider.REPLAY:
        if settings.record:
            raise ModelConfigError("TW_RECORD needs a live provider (gemini or openrouter)")
        replay = {
            role: ReplayClient(recordings(role), model=model) for role, model in models.items()
        }
        return ModelClients(clients=replay, models=models)

    # An unused AsyncClient holds no connections, so one left behind by an error here is harmless.
    owned = http_client is None
    http = http_client or httpx.AsyncClient(
        timeout=httpx.Timeout(settings.turn_timeout_s, connect=10.0)
    )
    live = {role: _live_client(settings, model, http) for role, model in models.items()}
    clients: dict[str, ModelClient] = dict(live)
    if settings.record:
        clients = {
            role: RecordingClient(client, recordings(role), model=models[role])
            for role, client in live.items()
        }
    return ModelClients(clients=clients, models=models, _http=http if owned else None)


def _live_client(settings: Settings, model: str, http: httpx.AsyncClient) -> OpenAICompatClient:
    cls: type[OpenAICompatClient]
    if settings.llm_provider is Provider.GEMINI:
        key, base_url, cls = settings.gemini_api_key, settings.gemini_base_url, GeminiClient
        name = "GEMINI_API_KEY"
    else:
        key, base_url = settings.openrouter_api_key, settings.openrouter_base_url
        cls, name = OpenRouterClient, "OPENROUTER_API_KEY"
    if key is None:
        raise ModelConfigError(f"{settings.llm_provider} needs {name}")
    return cls(model=model, api_key=key.get_secret_value(), base_url=base_url, http_client=http)
