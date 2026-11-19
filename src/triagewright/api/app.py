"""The FastAPI app (ADR-001 §9). A trusted internal ingress, bound to localhost by default.

Every response carries a server-generated ``X-Request-ID`` header, and every
error has the body ``{"error": {"code", "message", "request_id", "details"}}``.
Logs hold only the method, route template, status, latency, and request id:
never raw paths, bodies, headers, or tokens.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, SecretStr, ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from triagewright.api.services import (
    ApprovalNotFoundError,
    DecisionConflictError,
    Services,
    decide_approval,
    token_matches,
)
from triagewright.contracts.agents import ActionOutcome, FinalResponse
from triagewright.contracts.guardrails import ApprovalRequest
from triagewright.contracts.tickets import Ticket

log = logging.getLogger("triagewright.api")

REQUEST_ID_HEADER = "X-Request-ID"
_STATUS_CODES = {404: "not_found", 405: "method_not_allowed"}
_LOGGED_METHODS = frozenset({"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"})


class ApiError(Exception):
    def __init__(
        self, status: int, code: str, message: str, headers: dict[str, str] | None = None
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.headers = headers


class DecisionBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approved: bool
    note: str = ""


class Health(BaseModel):
    status: str = "ok"


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", ""))


def _error(
    request_id: str,
    status: int,
    code: str,
    message: str,
    *,
    details: list[dict[str, Any]] | None = None,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    body = {
        "error": {
            "code": code,
            "message": message,
            "request_id": request_id,
            "details": details or [],
        }
    }
    return JSONResponse(body, status_code=status, headers=headers)


def create_app(
    services: Services,
    *,
    approver_token: SecretStr | None,
    clock: Callable[[], datetime] = _utcnow,
) -> FastAPI:
    """Build the API on the given ports. ``approver_token`` unset disables decisions (503)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Lifespan shutdown runs on SIGINT and SIGTERM before uvicorn re-raises the signal,
        # and on the loop that served requests (it owns the model HTTP pool).
        try:
            yield
        finally:
            try:
                await services.aclose()
            finally:
                await asyncio.to_thread(services.close)  # bounded, blocking joins

    app = FastAPI(title="triagewright", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = uuid.uuid4().hex
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:
            # The exception text may quote customer content, so log only its type.
            log.error("unhandled %s request_id=%s", type(exc).__name__, request_id)
            response = _error(request_id, 500, "internal", "internal server error")
        response.headers[REQUEST_ID_HEADER] = request_id
        method = request.method if request.method in _LOGGED_METHODS else "<other>"
        route = getattr(request.scope.get("route"), "path", "<unmatched>")
        log.info(
            "%s %s %d %.1fms request_id=%s",
            method,
            route,
            response.status_code,
            (time.perf_counter() - started) * 1000,
            request_id,
        )
        return response

    @app.exception_handler(ApiError)
    async def api_error(request: Request, exc: ApiError) -> JSONResponse:
        return _error(_request_id(request), exc.status, exc.code, exc.message, headers=exc.headers)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _STATUS_CODES.get(exc.status_code, "http_error")
        return _error(
            _request_id(request), exc.status_code, code, str(exc.detail), headers=exc.headers
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Never echo the rejected input: it can hold customer PII.
        details = [
            {"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()
        ]
        return _error(
            _request_id(request), 422, "validation_error", "invalid request", details=details
        )

    def require_approver(authorization: Annotated[str | None, Header()] = None) -> None:
        if approver_token is None:
            raise ApiError(
                503, "decisions_disabled", "approval decisions are disabled (TW_APPROVER_TOKEN)"
            )
        scheme, _, presented = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not token_matches(presented.strip(), approver_token):
            raise ApiError(
                401,
                "unauthorized",
                "a valid approver bearer token is required",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @app.get("/healthz")
    async def healthz() -> Health:
        return Health()

    @app.post("/tickets")
    async def post_ticket(ticket: Ticket) -> FinalResponse:
        return await services.resolver.resolve(ticket)

    @app.get("/approvals")
    async def list_approvals() -> list[ApprovalRequest]:
        return services.approvals.list_pending()

    # The body is read by hand, after auth: FastAPI parses declared bodies before dependencies.
    @app.post(
        "/approvals/{approval_id}/decision",
        dependencies=[Depends(require_approver)],
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": DecisionBody.model_json_schema()}},
            }
        },
    )
    async def post_decision(approval_id: str, request: Request) -> ActionOutcome:
        try:
            body = DecisionBody.model_validate_json(await request.body())
        except ValidationError as exc:
            errors = [{**e, "loc": ("body", *e["loc"])} for e in exc.errors()]
            raise RequestValidationError(errors) from None
        try:
            return await decide_approval(
                services,
                approval_id,
                approved=body.approved,
                reviewer="api",
                note=body.note,
                now=clock(),
            )
        except ApprovalNotFoundError:
            raise ApiError(404, "not_found", "approval not found") from None
        except DecisionConflictError as exc:
            raise ApiError(409, "conflict", str(exc)) from None

    return app
