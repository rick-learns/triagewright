"""Render the README tour images from one local run, with no model and no API key.

The real wiring (in-process MCP tools, RAG, guardrails, SQLite approvals, JSONL
traces, the CLI and the HTTP API) runs with a scripted model in place of the
LLM. The eval excerpt comes from ``triagewright-evals run --stub``. Run from the
repo root:

    uv run python docs/assets/tour.py
"""

from __future__ import annotations

import asyncio
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import httpx
from pydantic import SecretStr
from rich.console import Console
from rich.syntax import Syntax
from rich.text import Text
from rich.tree import Tree
from typer.testing import CliRunner

from triagewright import wiring
from triagewright.api import create_app
from triagewright.api.services import Services
from triagewright.cli import create_cli
from triagewright.config import Settings
from triagewright.contracts.llm import ModelRequest, ModelResponse
from triagewright.testing import ScriptedModelClient, model_response, tool_use_block

ASSETS = Path(__file__).resolve().parent
TOKEN = "tour-approver-token"  # noqa: S105 - a throwaway token for this local run
WIDTH = 96
NOTE = "Defective within 30 days of delivery."

TICKET = {
    "ticket_id": "tkt_happy006",
    "customer_id": "cus_jun10",
    "channel": "web",
    "subject": "Smartwatch stopped working",
    "body": (
        "My Pulse Smartwatch from order ord_j1001 arrived two days ago and it already won't "
        "turn on, even after charging overnight. I'd like my $199.00 back."
    ),
    "created_at": "2026-06-15T07:55:00Z",
}


def call(name: str, arguments: dict[str, Any] | None = None) -> ModelResponse:
    return model_response(tool_use_block(name, arguments or {}))


def script() -> list[ModelResponse]:
    """What a model would answer at each step of eval case happy-006."""
    return [
        call(
            "submit_triage",
            {
                "category": "refund_request",
                "intent": "Refund a smartwatch that stopped working",
                "urgency": "normal",
                "needs_human": False,
                "needs_human_reasons": [],
                "injection_suspected": False,
                "refund_requested": True,
                "confidence": 0.92,
            },
        ),
        call("search_kb", {"query": "refund policy for an item that stopped working"}),
        call("action"),
        call("get_order", {"order_id": "ord_j1001"}),
        call(
            "propose_action",
            {
                "tool": "issue_refund",
                "arguments": {
                    "order_id": "ord_j1001",
                    "amount_cents": 19900,
                    "reason": "Stopped working two days after delivery",
                },
                "rationale": "Delivered 2 days ago, inside the 30-day refund window.",
            },
        ),
        call("submit_action", {"summary": "Proposed a $199.00 refund for ord_j1001."}),
        call(
            "finalize",
            {
                "status": "resolved",
                "reply": (
                    "Sorry your Pulse Smartwatch stopped working. I've requested a $199.00 "
                    "refund for order ord_j1001. A team member reviews every refund, usually "
                    "within 1 business day, and the money then reaches your original payment "
                    "method in 5 to 10 business days."
                ),
                "citations": [
                    {
                        "chunk_id": "kb_refund_policy#0",
                        "quote": "Delivered orders can be refunded within 30 days of the "
                        "delivery date.",
                    }
                ],
            },
        ),
        call(
            "submit_escalation",
            {
                "escalate": False,
                "reasons": [],
                "queue": None,
                "priority": "normal",
                "handoff_summary": "",
            },
        ),
    ]


class Scripted:
    """Scripted responses, stamped with the requested model so the real price table applies."""

    def __init__(self) -> None:
        self.inner = ScriptedModelClient(script())

    async def complete(self, request: ModelRequest) -> ModelResponse:
        response = await self.inner.complete(request)
        return response.model_copy(update={"model": request.model})


def console() -> Console:
    return Console(
        record=True,
        width=WIDTH,
        file=io.StringIO(),
        force_terminal=True,
        color_system="truecolor",
    )


def save(out: Console, name: str, title: str) -> None:
    out.save_svg(str(ASSETS / name), title=title)
    print(f"wrote docs/assets/{name}")


def show_json(out: Console, data: Any) -> None:
    out.print(Syntax(json.dumps(data, indent=2), "json", theme="ansi_dark", word_wrap=True))


def prompt(out: Console, command: str) -> None:
    out.print(Text("$ ", style="bold green") + Text(command, style="bold"))


def resolve(settings: Settings, services: Services, tmp: Path) -> str:
    ticket = tmp / "ticket.json"
    ticket.write_text(json.dumps(TICKET), encoding="utf-8")
    cli = create_cli(services_loader=lambda _: services, settings_loader=lambda: settings)
    result = CliRunner().invoke(cli, ["resolve", str(ticket)])
    assert result.exit_code == 0, result.output
    response = json.loads(result.output)
    assert response["status"] == "pending_approval", response

    out = console()
    prompt(out, "triagewright resolve ticket.json")
    # Trimmed for the image: drop the long fields, keep the shape.
    for cite in response["citations"]:
        del cite["chunk_id"], cite["title"]
    for action in response["actions"]:
        del action["action"]["rationale"], action["verdict"]["reasons"], action["attempts"]
        eligibility = action["eligibility"]
        action["eligibility"] = {k: eligibility[k] for k in ["eligible", "window_ends_at"]}
        action.pop("result", None)
    keep = ["status", "reply", "citations", "actions", "escalation"]
    show_json(out, {k: response[k] for k in keep})
    save(out, "tour-resolve.svg", "triagewright resolve")
    approval_id: str = response["actions"][0]["approval_id"]
    return approval_id


def approve(settings: Settings, services: Services, approval_id: str) -> None:
    app = create_app(services, approver_token=settings.approver_token)

    async def run() -> tuple[httpx.Response, httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://tour") as http:
            pending = await http.get("/approvals")
            decision = await http.post(
                f"/approvals/{approval_id}/decision",
                headers={"Authorization": f"Bearer {TOKEN}"},
                json={"approved": True, "note": NOTE},
            )
        await services.aclose()
        return pending, decision

    try:
        pending, decision = asyncio.run(run())
    finally:
        services.close()
    assert pending.status_code == 200, pending.text
    assert decision.status_code == 200, decision.text

    out = console()
    prompt(out, "curl -s localhost:8080/approvals")
    (first,) = pending.json()
    show_json(out, {k: first[k] for k in ["approval_id", "tool", "arguments", "status"]})
    out.print()
    prompt(
        out,
        f"curl -s -X POST localhost:8080/approvals/{approval_id}/decision \\\n"
        '    -H "Authorization: Bearer $TW_APPROVER_TOKEN" \\\n'
        f'    -d \'{{"approved": true, "note": "{NOTE}"}}\'',
    )
    outcome = decision.json()
    show_json(out, {k: outcome[k] for k in ["approval_id", "status", "result"] if k in outcome})
    save(out, "tour-approve.svg", "approve over the HTTP API")


def trace(trace_dir: Path) -> None:
    events = [
        json.loads(line)
        for path in sorted(trace_dir.glob("*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    request = next(e for e in events if e["parent_span_id"] is None and e["name"] == "request")
    spans = [e for e in events if e["request_id"] == request["request_id"]]
    children: dict[str | None, list[dict[str, Any]]] = {}
    for span in sorted(spans, key=lambda s: s["ts"]):
        children.setdefault(span["parent_span_id"], []).append(span)

    shown = {
        "status": "status",
        "tw.purpose": "purpose",
        "tw.cost_usd": "cost_usd",
        "caller": "caller",
        "tw.tool.is_error": "error",
        "source": "source",
        "injection_suspected": "injection",
        "tool": "tool",
    }

    def label(span: dict[str, Any]) -> Text:
        attrs = span["attributes"]
        bits = [
            f"{shown[k]}={attrs[k]}"
            for k in shown
            if k in attrs and span["name"] != f"tool.{attrs[k]}"  # tool.search_kb, tool=...
        ]
        return Text(span["name"], style="bold cyan") + Text("  " + " ".join(bits), style="dim")

    root = Tree(label(request))

    def grow(node: Tree, span: dict[str, Any]) -> None:
        for child in children.get(span["span_id"], []):
            if not child["name"].startswith("model."):  # the model: span has the usage
                grow(node.add(label(child)), child)

    grow(root, request)
    out = console()
    out.print(
        "# var/traces/<date>.jsonl: the spans of one request, drawn as a tree",
        style="dim",
        highlight=False,
    )
    out.print(root)
    save(out, "tour-trace.svg", "trace excerpt")


def evals(tmp: Path) -> None:
    report = tmp / "report.md"
    cmd = [
        "triagewright-evals",
        "run",
        "--stub",
        "--case",
        "escal-00[1-3]",
        "--report",
        str(report),
        "--json",
        str(tmp / "report.json"),
        "--results",
        str(tmp / "results.jsonl"),
    ]
    done = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
    assert done.returncode == 0, done.stderr
    lines = report.read_text(encoding="utf-8").splitlines()
    summary = lines[lines.index("## Summary") : lines.index("## By category")]
    failures = lines[lines.index("## Failures") :]
    out = console()
    prompt(out, "triagewright-evals run --stub --case 'escal-00[1-3]'")
    out.print(done.stdout.split("; report")[0], style="dim")  # drop the temp paths
    out.print()
    out.print(
        Syntax("\n".join([*summary, *failures]), "markdown", theme="ansi_dark", word_wrap=True)
    )
    save(out, "tour-evals.svg", "eval report excerpt")


def main() -> int:
    with tempfile.TemporaryDirectory() as name:
        tmp = Path(name)
        settings = Settings(
            _env_file=None,
            db_path=tmp / "world.sqlite",
            approvals_db_path=tmp / "approvals.sqlite",
            trace_exporter="jsonl",
            trace_dir=tmp / "traces",
            approver_token=SecretStr(TOKEN),
        )
        services = wiring.build_services(settings, model_factory=lambda _: Scripted())
        try:
            approval_id = resolve(settings, services, tmp)
        finally:
            services.close()  # flush the trace
        trace(settings.trace_dir)
        # The API owns a fresh bundle over the same approvals database.
        approve(
            settings,
            wiring.build_services(settings, model_factory=lambda _: Scripted()),
            approval_id,
        )
        evals(tmp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
