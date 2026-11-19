"""Model clients for the agent: one HTTP pool and pacer per process, routed by purpose."""

from __future__ import annotations

from collections.abc import Callable

import httpx

from triagewright.config import Settings
from triagewright.contracts.llm import ModelRequest, ModelResponse, Provider
from triagewright.contracts.ports import ModelClient
from triagewright.llm import ModelClients, ModelConfigError, build_model_clients
from triagewright.llm.ratelimit import RateLimits

ModelFactory = Callable[[str | None], ModelClient]  # case id (None outside evals) -> client


class RoleRouter:
    """One ``ModelClient`` over the per-role clients: each request goes to its purpose's client."""

    def __init__(self, clients: ModelClients) -> None:
        self.clients = clients

    async def complete(self, request: ModelRequest) -> ModelResponse:
        try:
            client = self.clients.for_role(request.purpose)
        except KeyError:
            raise ModelConfigError(f"no model client for purpose {request.purpose!r}") from None
        return await client.complete(request)


class _SharedHttp:
    """One HTTP pool for every live client a factory builds; none in replay."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: httpx.AsyncClient | None = None
        # Free-tier quotas are per model and project, so every client shares one pacer.
        self.limits = RateLimits.shared(settings.requests_per_minute(), quota=settings.quota_id())

    def get(self) -> httpx.AsyncClient | None:
        if self._settings.llm_provider is Provider.REPLAY:
            return None
        if self._client is None:
            timeout = httpx.Timeout(self._settings.turn_timeout_s, connect=10.0)
            self._client = httpx.AsyncClient(timeout=timeout)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            client, self._client = self._client, None
            await client.aclose()


def _model_factory(settings: Settings, http: _SharedHttp | None = None) -> ModelFactory:
    http = http or _SharedHttp(settings)

    def build(case_id: str | None) -> ModelClient:
        clients = build_model_clients(
            settings, case_id=case_id, http_client=http.get(), limits=http.limits
        )
        return RoleRouter(clients)

    return build
