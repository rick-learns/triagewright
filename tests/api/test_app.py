from __future__ import annotations

import dataclasses
import logging
from typing import Any

import pytest
import uvicorn
from fastapi.testclient import TestClient
from pydantic import SecretStr

import triagewright.api as api
from tests.api.stack import TOKEN, Stack
from triagewright.api import create_app
from triagewright.api.services import Services, WiringUnavailableError, load_services
from triagewright.config import Settings
from triagewright.contracts.agents import ActionOutcome, FinalResponse, ResolutionStatus
from triagewright.contracts.guardrails import ApprovalRequest, ApprovalStatus
from triagewright.contracts.tickets import Ticket
from triagewright.testing import sample_ticket, seed_as_of, seed_world

AUTH = {"Authorization": f"Bearer {TOKEN}"}


def client(stack: Stack, token: str | None = TOKEN) -> TestClient:
    app = create_app(
        stack.services,
        approver_token=SecretStr(token) if token is not None else None,
        clock=seed_as_of,
    )
    return TestClient(app, raise_server_exceptions=False)


def decide(c: TestClient, approval_id: str, approved: bool = True) -> tuple[int, Any]:
    r = c.post(f"/approvals/{approval_id}/decision", json={"approved": approved}, headers=AUTH)
    return r.status_code, r.json()


def status(stack: Stack) -> ApprovalStatus:
    stored = stack.approvals.get(stack.approval_id)
    assert stored is not None
    return stored.status


def ticket_json(ticket: Ticket | None = None) -> dict[str, object]:
    return (ticket or sample_ticket()).model_dump(mode="json")


def test_healthz_and_request_ids(stack: Stack, caplog: pytest.LogCaptureFixture) -> None:
    c = client(stack)
    with caplog.at_level(logging.INFO, logger="triagewright.api"):
        r = c.get("/healthz", headers={"X-Request-ID": "555-0100"})
        c.get("/cus_ava01/4111111111111111")
        c.request("CALL5550100", "/healthz")
    assert r.status_code == 200 and r.json() == {"status": "ok"}
    assert len(r.headers["X-Request-ID"]) == 32  # server-generated, never the caller's
    assert "GET /healthz 200" in caplog.text and "GET <unmatched> 404" in caplog.text
    assert "<other> /healthz" in caplog.text and "5550100" not in caplog.text
    assert "555-0100" not in caplog.text and "4111111111111111" not in caplog.text


def test_post_ticket_returns_the_final_response(stack: Stack) -> None:
    r = client(stack).post("/tickets", json=ticket_json())
    assert r.status_code == 200
    final = FinalResponse.model_validate(r.json())
    assert final.ticket_id == "tkt_0001" and final.status is ResolutionStatus.ESCALATED


def test_invalid_ticket_is_a_structured_error_without_the_input(stack: Stack) -> None:
    body = {**ticket_json(), "ticket_id": "card-4111111111111111", "extra": "x"}
    r = client(stack).post("/tickets", json=body)
    assert r.status_code == 422
    error = r.json()["error"]
    assert error["code"] == "validation_error" and error["request_id"] == r.headers["X-Request-ID"]
    assert {tuple(d["loc"]) for d in error["details"]} == {
        ("body", "ticket_id"),
        ("body", "extra"),
    }
    assert "4111111111111111" not in r.text


def test_resolver_failure_is_a_500_that_logs_no_content(
    stack: Stack, caplog: pytest.LogCaptureFixture
) -> None:
    class Broken:
        async def resolve(self, ticket: Ticket) -> FinalResponse:
            raise RuntimeError(f"boom {ticket.body}")

    broken = Services(Broken(), stack.services.executor, stack.approvals)
    c = client(Stack(broken, stack.gateway, stack.approvals, stack.approval_id))
    with caplog.at_level(logging.INFO, logger="triagewright.api"):
        r = c.post("/tickets", json=ticket_json(sample_ticket("my email is a@b.example")))
    assert r.status_code == 500 and r.json()["error"]["code"] == "internal"
    assert r.json()["error"]["request_id"] == r.headers["X-Request-ID"]
    assert "RuntimeError" in caplog.text and "a@b.example" not in caplog.text


def test_lists_pending_approvals(stack: Stack) -> None:
    r = client(stack).get("/approvals")
    assert r.status_code == 200
    pending = [ApprovalRequest.model_validate(a) for a in r.json()]
    assert [a.approval_id for a in pending] == [stack.approval_id]
    assert pending[0].status is ApprovalStatus.PENDING


def test_decisions_are_disabled_without_a_configured_token(stack: Stack) -> None:
    code, body = decide(client(stack, token=None), stack.approval_id)
    assert code == 503 and body["error"]["code"] == "decisions_disabled"
    assert status(stack) is ApprovalStatus.PENDING


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer wrong-token"},
        {"Authorization": f"Basic {TOKEN}"},
        {"Authorization": "Bearer"},
        {"Authorization": TOKEN},
    ],
)
def test_wrong_or_missing_token_is_401(
    stack: Stack, headers: dict[str, str], caplog: pytest.LogCaptureFixture
) -> None:
    c = client(stack)
    with caplog.at_level(logging.INFO, logger="triagewright.api"):
        r = c.post(
            f"/approvals/{stack.approval_id}/decision", json={"approved": True}, headers=headers
        )
    assert r.status_code == 401 and r.json()["error"]["code"] == "unauthorized"
    assert r.headers["WWW-Authenticate"] == "Bearer"
    assert status(stack) is ApprovalStatus.PENDING
    assert TOKEN not in caplog.text


@pytest.mark.parametrize(
    ("token", "headers", "expected"),
    [
        (None, AUTH, 503),
        (TOKEN, {}, 401),
        (TOKEN, AUTH, 422),
    ],
)
def test_auth_is_checked_before_the_body(
    stack: Stack, token: str | None, headers: dict[str, str], expected: int
) -> None:
    url = f"/approvals/{stack.approval_id}/decision"
    c = client(stack, token=token)
    for content in ("{not json", '{"bogus": 1}'):
        r = c.post(url, content=content, headers={**headers, "Content-Type": "application/json"})
        assert r.status_code == expected
    if expected == 422:
        assert r.json()["error"]["details"][0]["loc"] == ["body", "approved"]
    assert status(stack) is ApprovalStatus.PENDING


def test_approve_executes_through_the_executor(stack: Stack) -> None:
    c = client(stack)
    code, body = decide(c, stack.approval_id)
    assert code == 200
    outcome = ActionOutcome.model_validate(body)
    assert outcome.status.value == "executed" and outcome.approval_id == stack.approval_id
    assert len(stack.gateway.refunds) == len(seed_world().refunds) + 1
    assert c.get("/approvals").json() == []
    code, body = decide(c, stack.approval_id)
    assert code == 409 and body["error"]["code"] == "conflict"
    assert len(stack.gateway.refunds) == len(seed_world().refunds) + 1


def test_reject_skips_and_a_conflicting_repeat_is_409(stack: Stack) -> None:
    c = client(stack)
    code, body = decide(c, stack.approval_id, approved=False)
    assert code == 200 and body["status"] == "skipped"
    code, body = decide(c, stack.approval_id, approved=True)
    assert code == 409 and "already rejected" in body["error"]["message"]
    assert status(stack) is ApprovalStatus.REJECTED
    assert stack.gateway.refunds == seed_world().refunds


def test_unknown_approval_is_404(stack: Stack) -> None:
    c = client(stack)
    for approval_id in ("apr_000000000000", "not-an-id"):
        code, body = decide(c, approval_id)
        assert code == 404 and body["error"]["code"] == "not_found"


def test_unknown_route_is_a_structured_404(stack: Stack) -> None:
    r = client(stack).get("/nope")
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"


def test_main_exits_with_a_helpful_message_in_replay_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api, "get_settings", lambda: Settings(_env_file=None))
    with pytest.raises(SystemExit, match="replay only replays eval-case recordings"):
        api.main()
    with pytest.raises(WiringUnavailableError):
        load_services(Settings(_env_file=None))


def test_main_serves_the_app_on_the_configured_host_and_port(
    monkeypatch: pytest.MonkeyPatch, stack: Stack
) -> None:
    settings = Settings(_env_file=None, api_host="127.0.0.1", api_port=9123)
    served: dict[str, object] = {}
    monkeypatch.setattr(api, "get_settings", lambda: settings)
    closed: list[str] = []

    async def serve(server: uvicorn.Server) -> None:
        served.update(host=server.config.host, port=server.config.port)
        served.update(access_log=server.config.access_log)

    async def aclose() -> None:
        closed.append("aclose")

    services = dataclasses.replace(
        stack.services, aclose=aclose, close=lambda: closed.append("close")
    )
    monkeypatch.setattr(api, "load_services", lambda s: services)
    monkeypatch.setattr(uvicorn.Server, "serve", serve)
    api.main()
    assert served["host"] == "127.0.0.1" and served["port"] == 9123
    assert served["access_log"] is False
    assert closed == ["aclose", "close"]
