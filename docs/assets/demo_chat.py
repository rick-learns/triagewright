"""Render the offline demo as the README's animated support console, demo-chat.svg.

Runs four of the demo's scenes (``demo_run.py``) on the real stack with a scripted model, then
draws what came back as a support console: the inbox, the conversation, and the agent's
activity as it happens. The customer types, the agent works through its steps, replies with
citations, and a reviewer approves the refund. Every reply, citation, tool call, and outcome
is from the run. CSS animation only, so GitHub plays it in an <img>. No model, no API key, no
network. Run from the repo root:

    uv run python docs/assets/demo_chat.py
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from html import escape
from pathlib import Path

import demo_run
import tour
from pydantic import SecretStr
from rich.console import Console

from triagewright import wiring
from triagewright.config import Settings
from triagewright.contracts.agents import ActionOutcome, FinalResponse
from triagewright.contracts.common import JsonValue
from triagewright.contracts.tickets import Ticket
from triagewright.guardrails import SqliteApprovalStore

W, H = 1000, 640
TOP = 40  # window title bar
INBOX_W, CHAT_W = 230, 430  # the activity panel takes the rest
CHAT_X, ACT_X = INBOX_W, INBOX_W + CHAT_W
COMPOSE_Y = H - 64  # the message box at the bottom of the conversation
BUBBLE_W = 330
LINE = 19  # line height in a bubble
SANS = "ui-sans-serif,-apple-system,'Segoe UI',Helvetica,Arial,sans-serif"
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"
BG, PANEL, EDGE, TEXT, MUTED = "#0d1117", "#161b22", "#30363d", "#e6edf3", "#8b949e"
TONES = {"ok": "#3fb950", "warn": "#d29922", "block": "#f85149", "info": "#2f81f7"}
GLYPHS = {"ok": "✓", "warn": "!", "block": "✕", "info": "•"}

# Pacing, in seconds: slow enough to read along.
TYPE_CPS, TYPE_MAX = 40.0, 4.0  # the customer typing
SEND_GAP, STEP_S, LINE_S, CITE_S, HOLD_S = 0.8, 1.1, 0.35, 0.9, 4.5
FADE = 0.3
TOOLS = {"search_kb": "BM25 + embeddings over the KB", "get_order": "order lookup over MCP"}


@dataclass
class Step:
    """One line in the activity panel."""

    label: str
    detail: str
    tone: str = "ok"


@dataclass
class Card:
    """The outcome under the reply: what code did, as the console shows it."""

    tone: str
    title: str
    lines: list[str]
    approve: str | None = None  # the line shown once a reviewer approves


@dataclass
class Scene:
    subject: str
    who: str
    body: str
    steps: list[Step]
    reply: str
    citations: list[tuple[str, str]]
    card: Card
    pill: tuple[str, str]  # the inbox status once the ticket settles: text, tone
    after_approval: list[Step] = field(default_factory=list)


# --- the run -------------------------------------------------------------------------------


def money(cents: int) -> str:
    return f"${cents / 100:,.2f}"


def int_arg(args: Mapping[str, JsonValue], key: str) -> int:
    value = args[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"expected int for {key!r}, got {type(value).__name__}")
    return value


def opening(response: FinalResponse) -> list[Step]:
    triage = response.triage
    assert triage is not None, response
    injected = triage.injection_suspected
    steps = [
        Step("redact PII", "before any model sees the ticket"),
        Step(
            "screen for injection",
            "instructions planted in the ticket" if injected else "nothing planted",
            "block" if injected else "ok",
        ),
        Step(
            "triage",
            triage.category.value + (" · refund requested" if triage.refund_requested else ""),
            "info",
        ),
    ]
    return steps


def tool_steps(names: Sequence[tuple[str, str | None]]) -> list[Step]:
    steps: list[Step] = []
    attempts = sum(1 for _, error in names if error)
    tried = 0
    for name, error in names:
        if error:
            tried += 1
            steps.append(Step(name, f"{error} (attempt {tried} of {attempts})", "warn"))
        elif name == "create_escalation":
            steps.append(Step(name, "handoff stored over MCP", "info"))
        else:
            steps.append(Step(name, TOOLS.get(name, "tool call over MCP"), "ok"))
    return steps


def blocked(outcome: ActionOutcome) -> Step:
    rules = ", ".join(outcome.verdict.rule_ids) if outcome.verdict else "policy"
    amount = money(int_arg(outcome.action.arguments, "amount_cents"))
    return Step(f"{outcome.action.tool.value} {amount}", f"blocked by code: {rules}", "block")


def eval_scene(
    settings: Settings, case_id: str
) -> tuple[FinalResponse, list[tuple[str, str | None]]]:
    response, stack = demo_run.run_case(settings, demo_run.CASES[case_id])
    calls = [
        (c.tool.value, c.result.error.category.value if c.result.error else None)
        for c in stack.tool_calls()
    ]
    return response, calls


def who(ticket: Ticket) -> str:
    return f"{ticket.channel.value} · {ticket.customer_id or 'not signed in'}"


def grounded(settings: Settings) -> Scene:
    case = demo_run.CASES["happy-007"]
    response, calls = eval_scene(settings, case.id)
    assert response.status == "resolved" and response.citations, response
    steps = opening(response) + tool_steps(calls)
    steps.append(Step("check citations", f"{len(response.citations)} quotes found verbatim"))
    return Scene(
        subject=case.ticket.subject or case.id,
        who=who(case.ticket),
        body=case.ticket.body,
        steps=steps,
        reply=response.reply,
        citations=[(c.title, c.quote) for c in response.citations],
        card=Card("ok", "Resolved", ["Answered from the knowledge base. No handoff needed."]),
        pill=("resolved", "ok"),
    )


def refund(settings: Settings) -> Scene:
    ticket = Ticket.model_validate(tour.TICKET)
    services = wiring.build_services(settings, model_factory=lambda _: tour.Scripted())
    try:
        response = asyncio.run(services.resolver.resolve(ticket))
    finally:
        services.close()  # flush the trace
    assert response.status == "pending_approval", response
    outcome = response.actions[0]
    assert outcome.approval_id is not None and outcome.eligibility is not None, outcome

    # The tools this request called, from its trace.
    spans = [
        json.loads(line)
        for path in sorted(settings.trace_dir.glob("*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    names: list[tuple[str, str | None]] = [
        (str(s["name"]).removeprefix("tool."), None) for s in spans if s["name"].startswith("tool.")
    ]
    # Spans are written as they end; tools end in the order they ran.
    eligibility = outcome.eligibility
    amount = money(int_arg(outcome.action.arguments, "amount_cents"))
    order = str(outcome.action.arguments["order_id"])
    steps = opening(response) + tool_steps(names)
    steps.append(
        Step(
            "refund policy",
            f"eligible: window ends {eligibility.window_ends_at:%b %-d}"
            if eligibility.eligible
            else "not eligible",
            "ok" if eligibility.eligible else "block",
        )
    )
    steps.append(Step(f"issue_refund {amount}", "held: waits for a human approval", "warn"))

    # A reviewer approves it over the real HTTP API.
    tour.approve(
        settings,
        wiring.build_services(settings, model_factory=lambda _: tour.Scripted()),
        outcome.approval_id,
    )
    store = SqliteApprovalStore(settings.approvals_db_path)
    try:
        approval = store.get(outcome.approval_id)
    finally:
        store.close()
    assert approval is not None and approval.status == "executed", approval

    return Scene(
        subject=ticket.subject or ticket.ticket_id,
        who=who(ticket),
        body=ticket.body,
        steps=steps,
        reply=response.reply,
        citations=[(c.title, c.quote) for c in response.citations],
        card=Card(
            "warn",
            f"Refund {amount} · {order}",
            ["Inside the 30-day window. Waiting for a reviewer."],
            approve=f"Approved by a reviewer · refund {approval.status.value}",
        ),
        pill=("refunded", "ok"),
        after_approval=[
            Step("reviewer approves", "POST /approvals/{id}/decision", "info"),
            Step(f"issue_refund {amount}", f"{approval.status.value} with the stored arguments"),
        ],
    )


def injection(settings: Settings) -> Scene:
    case = demo_run.CASES["inject-001"]
    response, calls = eval_scene(settings, case.id)
    assert response.escalation is not None and response.escalation.queue == "security", response
    assert [a.status for a in response.actions] == ["blocked"], response
    steps = opening(response) + [blocked(a) for a in response.actions] + tool_steps(calls)
    outcome = response.actions[0]
    return Scene(
        subject=case.ticket.subject or case.id,
        who=who(case.ticket),
        body=case.ticket.body,
        steps=steps,
        reply=response.reply,
        citations=[],
        card=Card(
            "block",
            f"Blocked: {outcome.action.tool.value} "
            f"{money(int_arg(outcome.action.arguments, 'amount_cents'))}",
            [
                "The model proposed it; code refused. Nothing awaits approval.",
                f"Handed to the {response.escalation.queue.value} queue · {response.handoff_id}",
            ],
        ),
        pill=("security", "block"),
    )


def tool_failure(settings: Settings) -> Scene:
    case = demo_run.CASES["toolfail-003"]
    response, calls = eval_scene(settings, case.id)
    handoff = response.escalation
    assert handoff is not None and handoff.queue == "tier1", response
    failed = sum(1 for _, error in calls if error)
    return Scene(
        subject=case.ticket.subject or case.id,
        who=who(case.ticket),
        body=case.ticket.body,
        steps=opening(response) + tool_steps(calls),
        reply=response.reply,
        citations=[],
        card=Card(
            "warn",
            f"Handed off to {handoff.queue.value}",
            [
                f"get_order failed {failed} times (a call and its retries).",
                f"Reason: {', '.join(r.value for r in handoff.reasons)} · {response.handoff_id}",
            ],
        ),
        pill=("tier1", "warn"),
    )


def record() -> list[Scene]:
    # Like the demo: ignore exported settings and keys, so the run stays offline and keyless.
    for var in [
        v for v in os.environ if v.upper().startswith("TW_") or v.upper().endswith("_API_KEY")
    ]:
        del os.environ[var]
    demo_run.out = Console(file=io.StringIO())  # the scenes' own printing isn't drawn
    tour.console = lambda: Console(file=io.StringIO(), record=True)
    tour.save = lambda *_: None
    with tempfile.TemporaryDirectory() as name:
        tmp = Path(name)
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
        cases = settings.model_copy(update={"trace_dir": tmp / "case-traces"})
        return [grounded(cases), refund(settings), injection(cases), tool_failure(cases)]


# --- drawing -------------------------------------------------------------------------------


def wrap(text: str, chars: int) -> list[str]:
    """Greedy word wrap; a blank line between paragraphs."""
    lines: list[str] = []
    for n, paragraph in enumerate(text.split("\n\n")):
        if n:
            lines.append("")
        line = ""
        for word in paragraph.split():
            if line and len(line) + 1 + len(word) > chars:
                lines.append(line)
                line = word
            else:
                line = f"{line} {word}" if line else word
        lines.append(line)
    return lines


class Timeline:
    """Collects one keyframe animation per element over a single loop."""

    def __init__(self) -> None:
        self.windows: list[tuple[float, float, str]] = []
        self.moves: list[str] = []
        self.total = 0.0

    def show(self, start: float, end: float, kind: str = "slide") -> str:
        self.windows.append((start, end, kind))
        return f"a{len(self.windows) - 1}"

    def css(self) -> str:
        t = self.total

        def pct(s: float) -> str:
            return f"{max(0.0, min(100.0, s / t * 100)):.3f}%"

        rules = []
        for i, (start, end, kind) in enumerate(self.windows):
            hidden = "opacity:0;transform:translateY(8px)" if kind == "slide" else "opacity:0"
            shown = "opacity:1;transform:none" if kind == "slide" else "opacity:1"
            end = min(end, t - FADE)
            rules.append(
                f"@keyframes k{i}{{0%,{pct(start)}{{{hidden}}}{pct(start + FADE)},{pct(end)}"
                f"{{{shown}}}{pct(end + FADE)},100%{{opacity:0}}}}"
                f".a{i}{{opacity:0;animation:k{i} {t:.2f}s linear infinite}}"
            )
        return "\n".join(rules + self.moves)

    def typing(self, start: float, seconds: float, width: float, chars: int) -> str:
        """A cover that slides off the text a few characters at a time, like typing."""
        i = len(self.windows) + len(self.moves)
        t = self.total
        a, b = start / t * 100, (start + seconds) / t * 100
        self.moves.append(
            f"@keyframes m{i}{{0%,{a:.3f}%{{transform:translateX(0);"
            f"animation-timing-function:steps({max(1, chars // 2)},end)}}"
            f"{b:.3f}%,100%{{transform:translateX({width:.0f}px)}}}}"
            f".m{i}{{animation:m{i} {t:.2f}s linear infinite}}"
        )
        return f"m{i}"

    def path(
        self, start: float, seconds: float, frm: tuple[float, float], to: tuple[float, float]
    ) -> str:
        i = len(self.windows) + len(self.moves)
        t = self.total
        a, b = start / t * 100, (start + seconds) / t * 100
        self.moves.append(
            f"@keyframes m{i}{{0%,{a:.3f}%{{transform:translate({frm[0]}px,{frm[1]}px);"
            f"animation-timing-function:ease-in-out}}"
            f"{b:.3f}%,100%{{transform:translate({to[0]}px,{to[1]}px)}}}}"
            f".m{i}{{animation:m{i} {t:.2f}s linear infinite}}"
        )
        return f"m{i}"


def text(x: float, y: float, s: str, size: float = 13, fill: str = TEXT, **attrs: str) -> str:
    font = attrs.pop("font", None)
    family = MONO if font == "mono" else SANS
    extra = "".join(f' {k.replace("_", "-")}="{v}"' for k, v in attrs.items())
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" '
        f'fill="{fill}"{extra}>{escape(s)}</text>'
    )


def el(tag: str, **attrs: object) -> str:
    """One empty SVG element; underscores in attribute names become hyphens."""
    pairs = "".join(f' {k.replace("_", "-")}="{v}"' for k, v in attrs.items())
    return f"<{tag}{pairs}/>"


def group(cls: str, body: str, x: float = 0, y: float = 0) -> str:
    return f'<g transform="translate({x:.1f},{y:.1f})"><g class="{cls}">{body}</g></g>'


def pill(x: float, y: float, label: str, tone: str) -> str:
    color = TONES.get(tone, MUTED)
    w = 8 + 7 * len(label)
    return (
        f'<rect x="{x - w:.1f}" y="{y - 13}" width="{w}" height="18" rx="9" fill="{color}" '
        f'fill-opacity="0.15" stroke="{color}" stroke-opacity="0.6"/>'
        + text(x - w / 2, y, label, 11, color, text_anchor="middle")
    )


def scene_length(scene: Scene) -> float:
    typed = min(TYPE_MAX, len(scene.body) / TYPE_CPS)
    reply = len(wrap(scene.reply, 40)) * LINE_S
    approve = 4.0 + len(scene.after_approval) * STEP_S if scene.card.approve else 0.0
    return (
        0.6
        + typed
        + SEND_GAP
        + len(scene.steps) * STEP_S
        + 0.6
        + reply
        + len(scene.citations) * CITE_S
        + 1.0
        + approve
        + HOLD_S
    )


def draw(scenes: list[Scene]) -> str:
    tl = Timeline()
    starts, t = [], 0.0
    for scene in scenes:
        starts.append(t)
        t += scene_length(scene)
    tl.total = t
    parts: list[str] = []

    # Inbox: one row per ticket, its status, and which one is open.
    parts.append(text(18, TOP + 30, "Inbox", 12, MUTED, font_weight="600", letter_spacing="0.08em"))
    for n, scene in enumerate(scenes):
        y = TOP + 48 + n * 70
        start, end = starts[n], starts[n] + scene_length(scene)
        settle = end - HOLD_S - 1.0
        row = (
            f'<rect x="8" y="0" width="{INBOX_W - 16}" height="60" rx="8" fill="#1f6feb" '
            f'fill-opacity="0.16" stroke="#2f81f7" stroke-opacity="0.5"/>'
        )
        parts.append(group(tl.show(start, end - FADE, "fade"), row, 0, y))
        parts.append(text(20, y + 24, scene.subject, 12.5, TEXT, font_weight="600"))
        parts.append(text(20, y + 44, scene.who, 11.5, MUTED))
        parts.append(group(tl.show(0, start, "fade"), pill(INBOX_W - 18, y + 44, "new", "none")))
        parts.append(
            group(tl.show(start, settle, "fade"), pill(INBOX_W - 18, y + 44, "working", "info"))
        )
        parts.append(group(tl.show(settle, t, "fade"), pill(INBOX_W - 18, y + 44, *scene.pill)))

    for n, scene in enumerate(scenes):
        parts.extend(draw_scene(tl, scene, starts[n]))
    return frame("".join(parts), tl.css())


def draw_scene(tl: Timeline, scene: Scene, start: float) -> Iterator[str]:
    end = start + scene_length(scene)
    whole = tl.show(start, end - FADE, "fade")
    # Conversation header.
    yield group(
        whole,
        text(CHAT_X + 20, TOP + 30, scene.subject, 15, TEXT, font_weight="600")
        + text(CHAT_X + 20, TOP + 50, scene.who, 12, MUTED),
    )

    # The customer types, then sends.
    t = start + 0.6
    typed = min(TYPE_MAX, len(scene.body) / TYPE_CPS)
    preview = " ".join(scene.body.split())
    box_w = CHAT_W - 40
    shown = preview if len(preview) <= 46 else preview[:45].rstrip() + "…"
    cover = tl.typing(t, typed, box_w, len(shown))
    yield group(
        tl.show(start, t + typed + 0.2, "fade"),
        '<g clip-path="url(#compose)">'
        + text(CHAT_X + 34, COMPOSE_Y + 30, shown, 13, TEXT)
        + f'<g class="{cover}"><rect x="{CHAT_X + 30}" y="{COMPOSE_Y + 10}" width="{box_w}" '
        f'height="32" fill="{PANEL}"/></g></g>',
    )
    t += typed + SEND_GAP

    # Customer bubble.
    y = TOP + 76
    lines = wrap(" ".join(scene.body.split()), 40)
    h = 16 + LINE * len(lines)
    body = f'<rect x="0" y="0" width="{BUBBLE_W}" height="{h}" rx="12" fill="#21262d"/>'
    body += "".join(text(14, 22 + LINE * i, line, 13) for i, line in enumerate(lines))
    yield group(tl.show(t - 0.4, end - FADE), body, CHAT_X + 20, y)
    y += h + 14

    # The agent works: a typing indicator in the chat, steps in the activity panel.
    work = len(scene.steps) * STEP_S
    dots = "".join(f'<circle cx="{18 + 12 * i}" cy="17" r="3.5" fill="{MUTED}"/>' for i in range(3))
    dots_box = (
        '<rect x="0" y="0" width="62" height="34" rx="12" fill="#1f6feb" fill-opacity="0.35"/>'
    )
    dots_label = text(72, 22, "triagewright is working…", 12, MUTED)
    yield group(
        tl.show(t, t + work + 0.3, "fade"),
        dots_box + dots + dots_label,
        CHAT_X + CHAT_W - 20 - BUBBLE_W,
        y,
    )
    yield from draw_steps(tl, scene.steps, t, end, 0)
    t += work + 0.6

    # Reply, line by line, then citations and the outcome card.
    lines = wrap(scene.reply, 40)
    h = 16 + LINE * len(lines)
    x = CHAT_X + CHAT_W - 20 - BUBBLE_W
    yield group(
        tl.show(t, end - FADE, "fade"),
        f'<rect x="0" y="0" width="{BUBBLE_W}" height="{h}" rx="12" fill="#1f6feb"/>',
        x,
        y,
    )
    for i, line in enumerate(lines):
        if line:
            yield group(
                tl.show(t + LINE_S * i, end - FADE, "fade"),
                text(14, 22 + LINE * i, line, 13, "#ffffff"),
                x,
                y,
            )
    t += LINE_S * len(lines)
    y += h + 10

    for title, quote in scene.citations:
        qlines = wrap(f"“{quote}”", 42)
        ch = 28 + 17 * len(qlines)
        cite_card = (
            el("rect", width=BUBBLE_W, height=ch, rx=8, fill=PANEL, stroke=EDGE)
            + el("rect", width=3, height=ch, fill=TONES["ok"])
            + text(12, 17, f"{title} · quote checked ✓", 11, TONES["ok"], font="mono")
            + "".join(
                text(12, 35 + 17 * i, q, 12.5, TEXT, font_style="italic")
                for i, q in enumerate(qlines)
            )
        )
        yield group(tl.show(t, end - FADE), cite_card, x, y)
        t += CITE_S
        y += ch + 8

    # Outcome card.
    card = scene.card
    color = TONES[card.tone]
    ch = 34 + 18 * len(card.lines) + (40 if card.approve else 0)
    cx = CHAT_X + 20
    cw = CHAT_W - 40
    t += 0.3
    body = (
        el(
            "rect",
            width=cw,
            height=ch,
            rx=10,
            fill=color,
            fill_opacity=0.08,
            stroke=color,
            stroke_opacity=0.7,
        )
        + text(14, 22, card.title, 13.5, color, font_weight="600")
        + "".join(text(14, 42 + 18 * i, line, 12.5, TEXT) for i, line in enumerate(card.lines))
    )
    yield group(tl.show(t, end - FADE), body, cx, y)
    if y + ch > COMPOSE_Y - 6:
        raise SystemExit(
            f"{scene.subject}: the conversation overflows by {y + ch - COMPOSE_Y + 6:.0f}px"
        )
    if card.approve:
        by = 42 + 18 * len(card.lines) - 6
        buttons = (
            f'<rect x="14" y="{by}" width="92" height="28" rx="6" fill="{TONES["ok"]}"/>'
            + text(60, by + 19, "Approve", 13, "#0d1117", text_anchor="middle", font_weight="600")
            + f'<rect x="114" y="{by}" width="80" height="28" rx="6" fill="none" stroke="{EDGE}"/>'
            + text(154, by + 19, "Reject", 13, MUTED, text_anchor="middle")
            + text(208, by + 19, "reviewer", 11.5, MUTED)
        )
        click = t + 2.6
        yield group(tl.show(t, click + 0.15, "fade"), buttons, cx, y)
        flash = el("rect", x=14, y=by, width=92, height=28, rx=6, fill="#fff", fill_opacity=0.5)
        yield group(tl.show(click - 0.15, click + 0.1, "fade"), flash, cx, y)
        done = el(
            "rect",
            x=10,
            y=by - 2,
            width=cw - 20,
            height=32,
            rx=6,
            fill=TONES["ok"],
            fill_opacity=0.15,
        ) + text(22, by + 19, f"✓ {card.approve}", 13, TONES["ok"], font_weight="600")
        yield group(tl.show(click + 0.1, end - FADE, "fade"), done, cx, y)
        # The reviewer's pointer.
        target = (cx + 60, y + by + 16)
        move = tl.path(t + 0.6, 1.8, (ACT_X - 40, COMPOSE_Y + 20), target)
        arrow = "M0,0 L0,17 L4.5,13 L8,21 L11,20 L7.5,12 L13,12 Z"
        pointer = el("path", d=arrow, fill="#fff", stroke=BG)
        yield group(tl.show(t + 0.3, click + 1.2, "fade"), f'<g class="{move}">{pointer}</g>')
        yield from draw_steps(tl, scene.after_approval, click + 0.4, end, len(scene.steps))


def draw_steps(tl: Timeline, steps: list[Step], t: float, end: float, first: int) -> Iterator[str]:
    for i, step in enumerate(steps):
        y = TOP + 76 + (first + i) * 44
        color = TONES[step.tone]
        body = (
            f'<circle cx="11" cy="11" r="10" fill="{color}" fill-opacity="0.18" stroke="{color}"/>'
            + text(11, 15.5, GLYPHS[step.tone], 11, color, text_anchor="middle", font_weight="700")
            + text(30, 11, step.label, 12.5, TEXT, font="mono")
            + text(30, 28, step.detail, 11.5, MUTED)
        )
        yield group(tl.show(t + STEP_S * i, end - FADE), body, ACT_X + 18, y)


def frame(content: str, css: str) -> str:
    chrome = (
        f'<rect width="{W}" height="{H}" rx="12" fill="{BG}" stroke="{EDGE}"/>'
        f'<rect x="0.5" y="0.5" width="{W - 1}" height="{TOP}" rx="12" fill="{PANEL}"/>'
        f'<rect x="0.5" y="{TOP - 12}" width="{W - 1}" height="12" fill="{PANEL}"/>'
        f'<line x1="0" y1="{TOP}" x2="{W}" y2="{TOP}" stroke="{EDGE}"/>'
        + "".join(
            f'<circle cx="{20 + 18 * i}" cy="{TOP / 2}" r="6" fill="{c}"/>'
            for i, c in enumerate(["#ff5f57", "#febc2e", "#28c840"])
        )
        + text(
            W / 2, TOP / 2 + 5, "triagewright · support console", 13, MUTED, text_anchor="middle"
        )
        + text(W - 16, TOP / 2 + 5, "offline run · scripted model", 11.5, MUTED, text_anchor="end")
        + f'<line x1="{CHAT_X}" y1="{TOP}" x2="{CHAT_X}" y2="{H}" stroke="{EDGE}"/>'
        + f'<line x1="{ACT_X}" y1="{TOP}" x2="{ACT_X}" y2="{H}" stroke="{EDGE}"/>'
        + el(
            "rect",
            x=ACT_X + 1,
            y=TOP + 1,
            width=W - ACT_X - 2,
            height=H - TOP - 2,
            fill=PANEL,
            fill_opacity=0.5,
        )
        + text(
            ACT_X + 18,
            TOP + 30,
            "Agent activity",
            12,
            MUTED,
            font_weight="600",
            letter_spacing="0.08em",
        )
        + text(ACT_X + 18, TOP + 50, "guardrails and tools, as they run", 11.5, MUTED)
        + el(
            "rect",
            x=CHAT_X + 20,
            y=COMPOSE_Y + 8,
            width=CHAT_W - 40,
            height=36,
            rx=18,
            fill=PANEL,
            stroke=EDGE,
        )
        + f'<clipPath id="compose"><rect x="{CHAT_X + 21}" y="{COMPOSE_Y + 9}" '
        f'width="{CHAT_W - 42}" height="34" rx="17"/></clipPath>'
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
        f'role="img" aria-label="triagewright support console demo">'
        f"<style>{css}</style>{chrome}{content}</svg>\n"
    )


def main() -> None:
    scenes = record()
    svg = draw(scenes)
    path = tour.ASSETS / "demo-chat.svg"
    path.write_text(svg, encoding="utf-8")
    seconds = sum(scene_length(s) for s in scenes)
    print(f"wrote docs/assets/{path.name} ({seconds:.0f} s loop, {len(svg) // 1024} KB)")


if __name__ == "__main__":
    main()
