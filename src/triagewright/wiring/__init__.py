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

Modules: ``models`` (model clients), ``coordinator``, ``gateway`` (out-of-process
MCP), ``services`` (API and CLI), ``evals`` (per-case stacks). The two zero-arg
entry points live here and read ``get_settings`` from this namespace.
"""

from __future__ import annotations

from pathlib import Path

from triagewright.config import get_settings as get_settings
from triagewright.contracts.ports import ModelClient
from triagewright.evals.judge import JudgeClientFactory
from triagewright.llm import ModelConfigError, build_model_clients, recording_path
from triagewright.llm.factory import JUDGE_ROLE
from triagewright.wiring.coordinator import _coordinator as _coordinator
from triagewright.wiring.coordinator import _mode as _mode
from triagewright.wiring.evals import CaseStack as CaseStack
from triagewright.wiring.evals import EvalStackFactory as EvalStackFactory
from triagewright.wiring.evals import memory_retriever as memory_retriever
from triagewright.wiring.gateway import BackgroundMcpGateway as BackgroundMcpGateway
from triagewright.wiring.gateway import _retrieve as _retrieve
from triagewright.wiring.gateway import logger as logger
from triagewright.wiring.models import ModelFactory as ModelFactory
from triagewright.wiring.models import RoleRouter as RoleRouter
from triagewright.wiring.models import _model_factory as _model_factory
from triagewright.wiring.models import _SharedHttp as _SharedHttp
from triagewright.wiring.services import _approval_key as _approval_key
from triagewright.wiring.services import build_services as build_services


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
        clients = build_model_clients(
            settings, case_id=case_id, http_client=http.get(), limits=http.limits
        )
        return clients.for_role(JUDGE_ROLE)

    return build
