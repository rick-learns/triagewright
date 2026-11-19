"""Lane H: HTTP API. ``create_app`` builds it on injected ports; ``main`` serves it with uvicorn."""

from __future__ import annotations

import logging

import uvicorn

from triagewright.api.app import create_app
from triagewright.api.services import Services, WiringUnavailableError, load_services
from triagewright.config import get_settings

__all__ = ["Services", "create_app", "main"]


def main() -> None:
    settings = get_settings()
    try:
        services = load_services(settings)
    except WiringUnavailableError as exc:
        raise SystemExit(f"triagewright-api: {exc}") from None
    logging.basicConfig(level=logging.INFO)
    app = create_app(services, approver_token=settings.approver_token)
    # uvicorn's access log records client addresses; the app logs its own request lines.
    uvicorn.run(app, host=settings.api_host, port=settings.api_port, access_log=False)
