"""The composition root: builds the real lanes from ``Settings`` (ADR-001, task X1).

- :func:`build_services`: the API and CLI bundle. ``mcp_transport=inprocess``
  runs the tool service in this process over ``db_path``; ``stdio`` and
  ``http`` reach a separate MCP server through :class:`BackgroundMcpGateway`.
  Approvals live in ``approvals_db_path``. A live provider is required:
  replay only has recordings for eval cases.
- :func:`case_runtime_factory`: the eval stack. Each case gets its own
  in-memory world on the case's clock and faults, its own approval store,
  and the case's recordings. The KB index is built once, in memory.
- :func:`judge_client_factory`: the eval judge's client.

The mock world's clock is pinned to the seed's ``as_of`` (the case's in
evals), so the executor's policy clock is too.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import logging
import secrets
import threading
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import anyio
import httpx

from triagewright.agent.coordinator import PURPOSES, Coordinator
from triagewright.api.services import Services, WiringUnavailableError
from triagewright.config import Settings, get_settings
from triagewright.contracts.common import JsonObject
from triagewright.contracts.evals import ToolFault
from triagewright.contracts.guardrails import ApprovalRequest
from triagewright.contracts.llm import ModelRequest, ModelResponse, Provider
from triagewright.contracts.ports import (
    ActionExecutor,
    Guardrails,
    ModelClient,
    Retriever,
    TelemetryFactory,
    TicketResolver,
    ToolGateway,
)
from triagewright.contracts.tools import (
    TOOL_SPECS,
    CallContext,
    ErrorCategory,
    ToolCallRecord,
    ToolName,
    ToolResult,
    ToolSpec,
)
from triagewright.contracts.world import WorldState
from triagewright.data import load_as_of, load_kb
from triagewright.evals.judge import JudgeClientFactory
from triagewright.guardrails import GuardedActionExecutor, RuleGuardrails, SqliteApprovalStore
from triagewright.llm import ModelClients, ModelConfigError, build_model_clients, recording_path
from triagewright.llm.factory import JUDGE_ROLE
from triagewright.mcp_server import InProcessToolGateway, McpToolGateway, build_service
from triagewright.observability import OTelTelemetryFactory
from triagewright.rag import HybridRetriever, KBIndex, make_embedder

logger = logging.getLogger(__name__)

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

    def get(self) -> httpx.AsyncClient | None:
        if self._settings.llm_provider is Provider.REPLAY:
            return None
        if self._client is None:
            timeout = httpx.Timeout(self._settings.turn_timeout_s, connect=10.0)
            self._client = httpx.AsyncClient(timeout=timeout)
        return self._client


def _retrieve(future: asyncio.Future[McpToolGateway]) -> None:
    if not future.cancelled():
        future.exception()


def _mode(settings: Settings) -> Literal["live", "replay"]:
    return "replay" if settings.llm_provider is Provider.REPLAY else "live"


def _coordinator(
    settings: Settings,
    *,
    model: ModelClient,
    gateway: ToolGateway,
    guardrails: Guardrails,
    executor: ActionExecutor,
    telemetry: TelemetryFactory,
) -> Coordinator:
    models = settings.models()
    return Coordinator(
        model=model,
        gateway=gateway,
        guardrails=guardrails,
        executor=executor,
        telemetry=telemetry,
        models={p: models[p] for p in PURPOSES},
        config=settings.loop_config(),
        mode=_mode(settings),
    )


def _model_factory(settings: Settings) -> ModelFactory:
    http = _SharedHttp(settings)

    def build(case_id: str | None) -> ModelClient:
        clients = build_model_clients(settings, case_id=case_id, http_client=http.get())
        return RoleRouter(clients)

    return build


def _approval_key(settings: Settings) -> bytes:
    """``TW_APPROVAL_KEY``, or an ephemeral key when only this process signs and checks."""
    if settings.approval_key is not None:
        return settings.approval_key.get_secret_value().encode()
    if settings.mcp_transport != "inprocess":  # Settings already refuses this
        raise WiringUnavailableError("TW_APPROVAL_KEY is required for an out-of-process MCP server")
    return secrets.token_bytes(32)


# ---- API and CLI -------------------------------------------------------------------


def build_services(
    settings: Settings,
    *,
    model_factory: ModelFactory | None = None,
    telemetry: TelemetryFactory | None = None,
) -> Services:
    """The real lanes for the API and CLI. ``model_factory`` and ``telemetry`` are for tests."""
    if model_factory is None and settings.llm_provider is Provider.REPLAY:
        raise WiringUnavailableError(
            "TW_LLM_PROVIDER=replay only replays eval-case recordings; set TW_LLM_PROVIDER "
            "to gemini or openrouter (free models) to resolve tickets"
        )
    key = _approval_key(settings)
    gateway: ToolGateway
    if settings.mcp_transport == "inprocess":
        gateway = InProcessToolGateway(build_service(settings, approval_key=key), log_calls=False)
    else:
        gateway = BackgroundMcpGateway(
            lambda: McpToolGateway.from_settings(settings, log_calls=False)
        )
    as_of = load_as_of(settings.seed_dir)
    guardrails = RuleGuardrails()
    approvals = SqliteApprovalStore(settings.approvals_db_path)
    executor = GuardedActionExecutor(
        gateway,
        guardrails,
        approvals,
        approval_key=key,
        clock=lambda: as_of,
        loop=settings.loop_config(),
    )
    tracing: OTelTelemetryFactory | None = None
    if telemetry is None:
        telemetry = tracing = OTelTelemetryFactory.from_settings(settings)
    resolver = _coordinator(
        settings,
        model=(model_factory or _model_factory(settings))(None),
        gateway=gateway,
        guardrails=guardrails,
        executor=executor,
        telemetry=telemetry,
    )

    def close() -> None:
        if isinstance(gateway, BackgroundMcpGateway):
            gateway.close()
        approvals.close()
        if tracing is not None:
            tracing.shutdown()  # flush spans

    return Services(resolver=resolver, executor=executor, approvals=approvals, close=close)


class BackgroundMcpGateway:
    """A ``ToolGateway`` that holds its MCP session on a private event-loop thread.

    Services are built synchronously and may be called from more than one event
    loop (the CLI runs one per command), but an MCP session must stay in the
    task that opened it. So one owner task on this thread opens it, and every
    call hops onto that loop. A failed or stalled connect and a lost session
    are retryable ``transient`` results; the next call reconnects.
    """

    def __init__(
        self,
        open_gateway: Callable[[], McpToolGateway],
        *,
        connect_timeout_s: float = 10.0,
        close_timeout_s: float = 5.0,
    ) -> None:
        self._open = open_gateway
        self._connect_timeout_s = connect_timeout_s
        self._close_timeout_s = close_timeout_s
        self._lock = threading.Lock()  # guards _accepting and _pending (caller threads)
        self._accepting = True
        self._pending: set[concurrent.futures.Future[ToolResult]] = set()
        # Loop-thread state:
        self._loop = asyncio.new_event_loop()
        self._session: asyncio.Future[McpToolGateway] | None = None
        self._owner: asyncio.Task[None] | None = None
        self._closing = asyncio.Event()
        self._thread = threading.Thread(target=self._loop.run_forever, name="mcp-gateway")
        self._thread.daemon = True
        self._thread.start()

    def list_tools(self) -> Sequence[ToolSpec]:
        return list(TOOL_SPECS.values())

    async def call(self, name: ToolName, arguments: JsonObject, ctx: CallContext) -> ToolResult:
        with self._lock:  # close() can't slip between the check and the submit
            if not self._accepting:
                return ToolResult.fail(name, ErrorCategory.TRANSIENT, "MCP gateway closed")
            future = asyncio.run_coroutine_threadsafe(
                self._call(name, dict(arguments), ctx), self._loop
            )
            self._pending.add(future)
        future.add_done_callback(self._settled)
        return await asyncio.wrap_future(future)  # cancelling the caller cancels the call

    def close(self) -> None:
        """Stop taking calls, end the session, settle in-flight calls, stop the thread.

        Idempotent. Each wait is bounded, so a stalled server can't hang shutdown.
        """
        with self._lock:
            if not self._accepting:
                return
            self._accepting = False
            pending = list(self._pending)
        shutdown = asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)
        try:
            shutdown.result(timeout=self._close_timeout_s)
        except (TimeoutError, concurrent.futures.CancelledError):
            shutdown.cancel()
        _, stuck = concurrent.futures.wait(pending, timeout=self._close_timeout_s)
        for future in stuck:
            future.cancel()
        concurrent.futures.wait(stuck, timeout=self._close_timeout_s)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=self._close_timeout_s)
        if not self._thread.is_alive():
            self._loop.close()

    def _settled(self, future: concurrent.futures.Future[ToolResult]) -> None:
        with self._lock:
            self._pending.discard(future)

    async def _call(self, name: ToolName, arguments: JsonObject, ctx: CallContext) -> ToolResult:
        try:
            session = self._connect()
            gateway = await asyncio.wait_for(asyncio.shield(session), self._connect_timeout_s)
        except TimeoutError:
            self._abandon(session)
            logger.warning("MCP connect timed out after %gs", self._connect_timeout_s)
            return ToolResult.fail(name, ErrorCategory.TRANSIENT, "MCP server unavailable")
        except Exception as exc:
            logger.warning("MCP connect failed: %s", type(exc).__name__)
            return ToolResult.fail(name, ErrorCategory.TRANSIENT, "MCP server unavailable")
        return await gateway.call(name, arguments, ctx)

    def _connect(self) -> asyncio.Future[McpToolGateway]:
        if self._closing.is_set():
            raise ConnectionError("gateway closing")
        session = self._session
        if session is None or (session.done() and (session.exception() or session.result().lost)):
            session = self._session = self._loop.create_future()
            session.add_done_callback(_retrieve)  # a failure nobody awaited isn't logged as lost
            self._owner = self._loop.create_task(self._own(session))
        return session

    def _abandon(self, session: asyncio.Future[McpToolGateway]) -> None:
        """A stalled connect: cancel it so the next call starts over."""
        if self._session is session and not session.done():
            self._session = None
            if self._owner is not None:
                self._owner.cancel()

    async def _own(self, ready: asyncio.Future[McpToolGateway]) -> None:
        try:
            async with self._open() as gateway:
                ready.set_result(gateway)
                async with anyio.create_task_group() as tg:

                    async def until(wait: Callable[[], Awaitable[object]]) -> None:
                        await wait()
                        tg.cancel_scope.cancel()

                    tg.start_soon(until, gateway.wait_lost)
                    tg.start_soon(until, self._closing.wait)
        except Exception as exc:
            if not ready.done():
                ready.set_exception(exc)
            else:
                logger.warning("MCP session closed with %s", type(exc).__name__)
        finally:
            if not ready.done():  # cancelled mid-connect
                ready.set_exception(ConnectionError("MCP connect cancelled"))

    async def _shutdown(self) -> None:
        self._closing.set()
        owner = self._owner
        if owner is None:
            return
        if self._session is not None and not self._session.done():
            owner.cancel()  # a stalled handshake never sees _closing
        with contextlib.suppress(asyncio.CancelledError):
            await owner


# ---- evals -------------------------------------------------------------------------


@dataclass(frozen=True)
class CaseStack:
    """One eval case's real stack. Satisfies ``ports.CaseRuntime``."""

    resolver: TicketResolver
    gateway: InProcessToolGateway
    approvals: SqliteApprovalStore

    def tool_calls(self) -> list[ToolCallRecord]:
        return list(self.gateway.calls)

    def ledger(self) -> WorldState:
        return self.gateway.ledger()

    def pending_approvals(self) -> list[ApprovalRequest]:
        return self.approvals.list_pending()


def memory_retriever(settings: Settings) -> HybridRetriever:
    """The KB at ``settings.kb_dir``, indexed in memory (evals never touch ``var/``)."""
    if not settings.kb_dir.is_dir():
        raise FileNotFoundError(f"KB directory not found: {settings.kb_dir}")
    index = KBIndex(":memory:", make_embedder(settings.embedder))
    index.ingest(load_kb(settings.kb_dir))
    return HybridRetriever(index)


class EvalStackFactory:
    """Builds a fresh :class:`CaseStack` per case. Satisfies ``ports.CaseRuntimeFactory``."""

    def __init__(
        self,
        settings: Settings,
        *,
        model_factory: ModelFactory | None = None,
        telemetry: TelemetryFactory | None = None,
        retriever: Retriever | None = None,
    ) -> None:
        self.settings = settings
        self._models = model_factory or _model_factory(settings)
        self._telemetry = telemetry or OTelTelemetryFactory.from_settings(settings)
        self._retriever = retriever or memory_retriever(settings)
        self._guardrails = RuleGuardrails()
        self._key = secrets.token_bytes(32)  # signs and checks within each case's stack

    def build(
        self, *, case_id: str, world: WorldState, as_of: datetime, faults: Sequence[ToolFault]
    ) -> CaseStack:
        gateway = InProcessToolGateway.for_world(
            world,
            as_of=as_of,
            approval_key=self._key,
            retriever=self._retriever,
            faults=faults,
            policy=self.settings.policy_config(),
        )
        approvals = SqliteApprovalStore()  # in memory
        executor = GuardedActionExecutor(
            gateway,
            self._guardrails,
            approvals,
            approval_key=self._key,
            clock=lambda: as_of,
            loop=self.settings.loop_config(),
        )
        resolver = _coordinator(
            self.settings,
            model=self._models(case_id),
            gateway=gateway,
            guardrails=self._guardrails,
            executor=executor,
            telemetry=self._telemetry,
        )
        return CaseStack(resolver=resolver, gateway=gateway, approvals=approvals)


def case_runtime_factory() -> EvalStackFactory:
    """``triagewright-evals run``'s default ``--factory``."""
    return EvalStackFactory(get_settings())


def judge_client_factory() -> JudgeClientFactory:
    """For ``triagewright-evals run --judge triagewright.wiring:judge_client_factory``."""
    settings = get_settings()
    http = _SharedHttp(settings)

    def build(*, case_id: str, recordings: Path) -> ModelClient:
        expected = recording_path(settings.recordings_dir, case_id, judge=True)
        if recordings != expected:
            raise ModelConfigError(f"judge recordings {recordings} are not {expected}")
        clients = build_model_clients(settings, case_id=case_id, http_client=http.get())
        return clients.for_role(JUDGE_ROLE)

    return build
