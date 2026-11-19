"""The offline presenter demo behind ``scripts/demo.sh``: no model, no API key, no network.

Each step runs the real wiring with a scripted model in place of the LLM, reusing the
README tour (``tour.py``) for the refund, its approval over the HTTP API, and the trace
tree. The other steps run one eval case each on its own isolated eval stack, with the
case's world and injected faults. Run from the repo root:

    uv run python docs/assets/demo_run.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import tour
from pydantic import SecretStr
from rich.console import Console
from rich.rule import Rule

from triagewright import wiring
from triagewright.config import Settings
from triagewright.contracts.agents import FinalResponse
from triagewright.contracts.evals import EvalCase
from triagewright.contracts.llm import ModelResponse
from triagewright.data import load_as_of, load_eval_cases, load_world, merge_world
from triagewright.guardrails import SqliteApprovalStore
from triagewright.testing import ScriptedModelClient

CASES = {case.id: case for case in load_eval_cases(tour.REPO / "evals" / "cases")}
call = tour.call
out = Console(width=tour.WIDTH, highlight=False)


def triage(category: str, intent: str, **extra: Any) -> ModelResponse:
    fields = {
        "category": category,
        "intent": intent,
        "urgency": "normal",
        "needs_human": False,
        "needs_human_reasons": [],
        "injection_suspected": False,
        "refund_requested": False,
        "confidence": 0.9,
    }
    return call("submit_triage", {**fields, **extra})


RETURNS = {"doc_id": "kb_returns", "chunk_id": "kb_returns#0", "title": "Returning an item"}
QUOTES = [
    "You can return most items within 30 days of delivery.",
    "We'll email you a prepaid shipping label.",
]
NO_ESCALATION = call("submit_escalation", {"escalate": False, "reasons": [], "queue": None})

# What a model would answer at each step of each case.
SCRIPTS: dict[str, list[ModelResponse]] = {
    "happy-007": [
        triage("policy_question", "Return window and who pays return shipping"),
        call("answer"),
        call("search_kb", {"query": "return window and return shipping label"}),
        call(
            "submit_answer",
            {
                "answer": "You can return most items within 30 days of delivery, and we email "
                "you a prepaid shipping label, so the return shipping is on us.",
                "citations": [{**RETURNS, "quote": q} for q in QUOTES],
                "grounded": True,
                "confidence": 0.9,
            },
        ),
        call(
            "finalize",
            {
                "status": "resolved",
                "reply": "You can return most items within 30 days of delivery. Start the return "
                "from Orders > Return item, and we'll email you a prepaid shipping label, so you "
                "don't pay for return shipping.",
                "citations": [{"chunk_id": RETURNS["chunk_id"], "quote": q} for q in QUOTES],
            },
        ),
        NO_ESCALATION,
    ],
    # The model falls for the injection and proposes the refund; code blocks it.
    "inject-001": [
        triage(
            "refund_request",
            "Refund $5000 on ord_a1001 to a pasted card",
            injection_suspected=True,
            refund_requested=True,
        ),
        call("action"),
        call(
            "propose_action",
            {
                "tool": "issue_refund",
                "arguments": {"order_id": "ord_a1001", "amount_cents": 500000, "reason": "admin"},
                "rationale": "The customer says the policy no longer applies.",
            },
        ),
        call("submit_action", {"summary": "Proposed a $5000.00 refund for ord_a1001."}),
        # No finalize: once a refund is proposed, code writes the reply (``_settled``).
        NO_ESCALATION,
    ],
    "toolfail-003": [
        triage("order_status", "Status of order ord_b2002"),
        call("get_order", {"order_id": "ord_b2002"}),
        call(
            "finalize",
            {
                "status": "resolved",
                "reply": "I couldn't load order ord_b2002 just now, so I've passed this to a "
                "colleague who will follow up with its status.",
            },
        ),
        NO_ESCALATION,
    ],
}


def scripted(responses: list[ModelResponse]) -> tour.Scripted:
    """The tour's scripted client (stamps the requested model, so prices apply) on a script."""
    client = tour.Scripted()
    client.inner = ScriptedModelClient(responses)
    return client


def step(n: int, title: str) -> None:
    out.print()
    out.print(Rule(f"[bold]{n}. {title}", align="left"))


def show(data: Any) -> None:
    out.print_json(json.dumps(data, default=lambda o: o.model_dump(mode="json")))


def run_case(settings: Settings, case: EvalCase) -> tuple[FinalResponse, wiring.CaseStack]:
    seed = settings.seed_dir
    stack = wiring.EvalStackFactory(
        settings, model_factory=lambda _: scripted(SCRIPTS[case.id])
    ).build(
        case_id=case.id,
        world=merge_world(load_world(seed), case.world),
        as_of=case.as_of or load_as_of(seed),
        faults=case.faults,
    )
    out.print(f"[dim]eval case {case.id}: {case.title}[/]")
    out.print(f"[bold]ticket:[/] {case.ticket.body.strip()}")
    return asyncio.run(stack.resolver.resolve(case.ticket)), stack


def calls(stack: wiring.CaseStack) -> list[str]:
    return [
        f"{c.tool.value} caller={c.caller}"
        + (f" error={c.result.error.category}" if c.result.error else "")
        for c in stack.tool_calls()
    ]


def escalation(response: FinalResponse) -> dict[str, Any] | None:
    handoff = response.escalation
    return None if handoff is None else {"queue": handoff.queue, "reasons": handoff.reasons}


def grounded(settings: Settings) -> None:
    step(1, "A grounded answer with citations")
    response, stack = run_case(settings, CASES["happy-007"])
    show(
        {
            "status": response.status,
            "reply": response.reply,
            "citations": [{"doc_id": c.doc_id, "quote": c.quote} for c in response.citations],
            "tool_calls": calls(stack),
        }
    )
    assert response.status == "resolved" and response.citations, response
    out.print("[dim]Quotes are checked verbatim against KB chunks this request retrieved.[/]")


def refund(settings: Settings, tmp: Path) -> None:
    step(2, "A refund held for approval, then approved over the HTTP API")
    out.print(f"[bold]ticket:[/] {tour.TICKET['body']}")
    services = wiring.build_services(settings, model_factory=lambda _: tour.Scripted())
    try:
        approval_id = tour.resolve(settings, services, tmp)
    finally:
        services.close()  # flush the trace
    out.print(
        "[dim]The same refund in the approvals queue, then a reviewer's decision, sent to the "
        "real FastAPI app in-process (no server needed):[/]"
    )
    # Like the tour, the API owns a fresh bundle over the same approvals database.
    api_services = wiring.build_services(settings, model_factory=lambda _: tour.Scripted())
    tour.approve(settings, api_services, approval_id)
    store = SqliteApprovalStore(settings.approvals_db_path)
    try:
        approval = store.get(approval_id)
    finally:
        store.close()
    assert approval is not None and approval.status == "executed", approval


def injection(settings: Settings) -> None:
    step(3, "A prompt-injection ticket goes to the security queue")
    response, stack = run_case(settings, CASES["inject-001"])
    show(
        {
            "status": response.status,
            "reply": response.reply,
            "actions": [
                {"tool": a.action.tool, "status": a.status, "verdict": a.verdict}
                for a in response.actions
            ],
            "escalation": escalation(response),
            "tool_calls": calls(stack),
            "pending_approvals": len(stack.pending_approvals()),
        }
    )
    assert response.escalation is not None and response.escalation.queue == "security", response
    assert [a.status for a in response.actions] == ["blocked"], response
    assert all(c.caller == "coordinator" for c in stack.tool_calls()), calls(stack)
    assert not stack.pending_approvals()
    out.print(
        "[dim]The model proposed the $5000 refund and code blocked it: issue_refund never ran "
        "and nothing awaits approval. The calls above are code's (caller=coordinator): the "
        "refund-policy lookup and the security handoff.[/]"
    )


def tool_failure(settings: Settings) -> None:
    step(4, "A tool that keeps failing escalates")
    response, stack = run_case(settings, CASES["toolfail-003"])
    show(
        {
            "status": response.status,
            "reply": response.reply,
            "escalation": escalation(response),
            "tool_calls": calls(stack),
        }
    )
    assert response.escalation is not None and response.escalation.queue == "tier1", response
    out.print("[dim]get_order timed out on all 3 attempts (a call, 2 retries): code escalates.[/]")


def trace(settings: Settings) -> None:
    step(5, "The trace of the refund request: tokens, cost, latency")
    events = [
        json.loads(line)
        for path in sorted(settings.trace_dir.glob("*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    root = next(e for e in events if e["parent_span_id"] is None and e["name"] == "request")
    out.print(f"[dim]var/traces/<date>.jsonl (a temp dir here), request {root['request_id']}:[/]")
    keys = [
        "tw.model_calls",
        "tw.tool_calls",
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens",
        "tw.cost_usd",
        "tw.latency_ms",
        "tw.estimated",
    ]
    totals = {k: root["attributes"][k] for k in keys}
    totals["tw.latency_ms"] = round(totals["tw.latency_ms"], 1)
    show(totals)
    out.print(
        "[dim]Offline, the tokens are the scripted replies' placeholder usage and the latency "
        "is estimated (tw.estimated). Cost is priced from the model each call names: the free "
        "defaults, so $0.[/]"
    )
    out.print("[dim]The same trace as a span tree (as in the README's tour-trace.svg):[/]")
    tour.trace(settings.trace_dir)


def main() -> int:
    # Like the tour: ignore exported settings and keys, so the demo stays offline and keyless.
    for var in [
        v for v in os.environ if v.upper().startswith("TW_") or v.upper().endswith("_API_KEY")
    ]:
        del os.environ[var]
    with tempfile.TemporaryDirectory() as name:
        tmp = Path(name)
        tour.console = lambda: out  # the tour's scenes print here instead of rendering SVGs
        tour.save = lambda *_: None
        settings = Settings(
            _env_file=None,
            mcp_transport="inprocess",
            embedder="hashing",
            db_path=tmp / "world.sqlite",
            approvals_db_path=tmp / "approvals.sqlite",
            trace_exporter="jsonl",
            trace_dir=tmp / "traces",
            approver_token=SecretStr(tour.TOKEN),
        )
        # The eval cases trace elsewhere, so the refund is the only request in its trace.
        cases = settings.model_copy(update={"trace_dir": tmp / "case-traces"})
        grounded(cases)
        refund(settings, tmp)
        injection(cases)
        tool_failure(cases)
        trace(settings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
