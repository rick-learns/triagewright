"""Model routing: the role router, the shared HTTP pool, and the shared pacer."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast

import httpx
import pytest

from tests.wiring.helpers import settings
from triagewright import wiring
from triagewright.config import Settings
from triagewright.contracts.llm import ModelRequest
from triagewright.llm import ModelConfigError, build_model_clients, recording_path
from triagewright.llm.openai_compat import GeminiClient

pytestmark = pytest.mark.anyio


def test_role_router_rejects_an_unknown_purpose(tmp_path: Path) -> None:
    clients = build_model_clients(settings(tmp_path), case_id="happy-001")
    router = wiring.RoleRouter(clients)
    request = ModelRequest.model_validate(
        {
            "model": "gemini-3.5-flash-lite",
            "system": "s",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "purpose": "nope",
        }
    )
    with pytest.raises(ModelConfigError, match="purpose 'nope'"):
        asyncio.run(router.complete(request))


def test_live_clients_share_one_http_pool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    live = settings(
        tmp_path,
        llm_provider="gemini",
        gemini_api_key="test-key-not-real",
        recordings_dir=tmp_path / "recordings",
    )
    monkeypatch.setattr(wiring, "get_settings", lambda: live)
    build = wiring.judge_client_factory()
    one = build(case_id="happy-001", recordings=tmp_path / "recordings" / "happy-001.judge.jsonl")
    two = build(case_id="happy-002", recordings=tmp_path / "recordings" / "happy-002.judge.jsonl")
    assert one.model == two.model == live.models()["judge"]  # type: ignore[attr-defined]
    assert one._http is two._http  # type: ignore[attr-defined]


def test_the_shared_http_pool_closes_on_the_loop_that_used_it() -> None:
    http = wiring._SharedHttp(Settings(_env_file=None, llm_provider="gemini"))

    async def use_then_close() -> httpx.AsyncClient:
        client = http.get()
        assert client is not None
        await http.aclose()
        return client

    assert asyncio.run(use_then_close()).is_closed
    asyncio.run(http.aclose())  # idempotent


def test_every_client_a_process_builds_shares_one_pacer(tmp_path: Path) -> None:
    live = settings(tmp_path, llm_provider="gemini", gemini_api_key="k" * 20)
    http = wiring._SharedHttp(live)
    assert http.limits.per_minute == 15
    factory = wiring._model_factory(live, http)
    a = cast(wiring.RoleRouter, factory("case-a"))
    b = cast(wiring.RoleRouter, factory("case-b"))
    pacers = [
        cast(GeminiClient, r.clients.for_role(role))._limiter
        for r in (a, b)
        for role in ("triage", "action")
    ]
    assert pacers[0] is not None and all(p is pacers[0] for p in pacers)


def test_the_eval_judge_shares_the_cases_pacer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live = settings(
        tmp_path,
        llm_provider="gemini",
        gemini_api_key="k" * 20,
        model_judge="gemini-3.5-flash-lite",
    )
    monkeypatch.setattr(wiring, "get_settings", lambda: live)
    cases = wiring._model_factory(live, wiring._SharedHttp(live))
    router = cast(wiring.RoleRouter, cases("case-a"))
    judge = wiring.judge_client_factory()(
        case_id="case-a", recordings=recording_path(live.recordings_dir, "case-a", judge=True)
    )
    case_pacer = cast(GeminiClient, router.clients.for_role("triage"))._limiter
    judge_pacer = cast(GeminiClient, judge)._limiter
    assert case_pacer is judge_pacer and case_pacer is not None
