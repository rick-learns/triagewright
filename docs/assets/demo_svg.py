"""Render the offline demo (``scripts/demo.sh``) as the README's animated terminal, demo.svg.

Runs demo_run.py's five steps on a recording console, then draws the output in a fixed-height
terminal window that reveals one line at a time, scrolls, and loops. CSS animation only, so
GitHub plays it in an <img>. No model, no API key, no network. Run from the repo root:

    uv run python docs/assets/demo_svg.py
"""

from __future__ import annotations

import io
import time
from html import escape

import demo_run
import tour
from rich.cells import cell_len
from rich.console import Console
from rich.style import Style
from rich.terminal_theme import SVG_EXPORT_THEME as THEME
from rich.text import Text

ROWS = 30  # visible terminal rows
LINE_H = 19.0
CHAR_W = 8.4  # one cell at a 14 px monospace font
PAD_X, TOP = 16.0, 44.0  # text inset; window chrome above the text
LINE_S, STEP_S, PROMPT_S, HOLD_S = 0.07, 3.0, 0.8, 6.0  # pauses: each line, before a step, a prompt


def record() -> list[Text]:
    out = Console(
        record=True,
        width=tour.WIDTH,
        height=ROWS,  # with both set, Rich keeps the size even under TERM=dumb
        file=io.StringIO(),
        force_terminal=True,
        color_system="truecolor",
        highlight=False,
    )
    demo_run.out = out  # demo_run's steps, and the tour scenes it reuses, print here
    out.print(Text("$ ", style="bold green") + Text("bash scripts/demo.sh", style="bold"))
    start = time.monotonic()
    demo_run.main()
    out.print()
    seconds = round(time.monotonic() - start)
    out.print(f"demo finished in {seconds} s, offline, with a scripted model")
    return Text.from_ansi(out.export_text(styles=True)).split("\n", allow_blank=True)


def pause_before(line: Text) -> float:
    plain = line.plain
    if plain[:2].rstrip(".").isdigit() and "──" in plain:  # a step's heading rule
        return STEP_S
    if plain.startswith("$ "):
        return PROMPT_S
    return LINE_S


def hex_color(style: Style, foreground: bool) -> str | None:
    color = style.color if foreground else style.bgcolor
    if color is None or color.is_default:
        return None
    return color.get_truecolor(THEME, foreground=foreground).hex


def draw_line(line: Text, row: int, console: Console) -> list[str]:
    y = row * LINE_H
    parts: list[str] = []
    line = line.copy()
    line.truncate(tour.WIDTH, overflow="ellipsis")  # JSON strings aren't wrapped
    col = 0
    for segment in line.render(console):
        cells = cell_len(segment.text)
        if not cells:
            continue
        style = segment.style or Style()
        x = PAD_X + col * CHAR_W
        width = cells * CHAR_W
        background = hex_color(style, foreground=False)
        if background:
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{LINE_H}" '
                f'fill="{background}"/>'
            )
        if segment.text.strip():
            attrs = [f'fill="{hex_color(style, foreground=True) or THEME.foreground_color.hex}"']
            if style.bold:
                attrs.append('font-weight="bold"')
            if style.dim:
                attrs.append('opacity="0.6"')
            if style.italic:
                attrs.append('font-style="italic"')
            parts.append(
                f'<text x="{x:.1f}" y="{y + LINE_H * 0.75:.1f}" textLength="{width:.1f}" '
                f'lengthAdjust="spacingAndGlyphs" {" ".join(attrs)}>{escape(segment.text)}</text>'
            )
        col += cells
    return parts


def keyframes(name: str, stops: list[tuple[float, float]], total: float) -> str:
    frames = "".join(f"{100 * t / total:.3f}%{{transform:translateY({v:.1f}px)}}" for t, v in stops)
    return f"@keyframes {name}{{{frames}100%{{transform:translateY({stops[-1][1]:.1f}px)}}}}"


def render(lines: list[Text]) -> str:
    console = Console(color_system="truecolor", width=tour.WIDTH, height=ROWS)
    clock = 0.0
    reveals: list[tuple[float, float]] = []  # (time, cover offset)
    scrolls: list[tuple[float, float]] = [(0.0, 0.0)]
    for shown, line in enumerate(lines, start=1):
        pause = pause_before(line)
        clock += PROMPT_S if pause == STEP_S and clock < STEP_S else pause  # step 1 starts at once
        reveals.append((clock, shown * LINE_H))
        offset = -max(0, shown - ROWS) * LINE_H
        if offset != scrolls[-1][1]:
            scrolls.append((clock, offset))
    total = clock + HOLD_S
    width = 2 * PAD_X + tour.WIDTH * CHAR_W
    view_h = ROWS * LINE_H
    height = TOP + view_h + 14
    content = len(lines) * LINE_H
    body = [part for row, line in enumerate(lines) for part in draw_line(line, row, console)]
    # Reduced motion: no animation, a static outline instead (the command, each step, the end).
    steps = [line for line in lines if pause_before(line) == STEP_S]
    outline = [lines[0], *steps, next(line for line in reversed(lines) if line.plain.strip())]
    summary = [part for row, line in enumerate(outline) for part in draw_line(line, row, console)]
    background = THEME.background_color.hex
    style = (
        "text{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,'Liberation Mono',"
        "monospace;font-size:14px;white-space:pre}"
        f"#scroll{{animation:scroll {total:.2f}s step-end infinite}}"
        f"#cover{{animation:cover {total:.2f}s step-end infinite}}"
        f"{keyframes('scroll', scrolls, total)}"
        f"{keyframes('cover', [(0.0, 0.0), *reveals], total)}"
        "#summary{display:none}"
        "@media (prefers-reduced-motion:reduce){#scroll{display:none}#summary{display:inline}}"
    )
    title = "bash scripts/demo.sh: the real stack, a scripted model in place of the LLM, offline"
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
            f'viewBox="0 0 {width:.0f} {height:.0f}" role="img" aria-label="{escape(title)}">',
            f"<title>{escape(title)}</title>",
            f"<style>{style}</style>",
            f'<rect width="100%" height="100%" rx="8" fill="{background}"/>',
            '<circle cx="20" cy="18" r="6" fill="#ff5f57"/>',
            '<circle cx="40" cy="18" r="6" fill="#febc2e"/>',
            '<circle cx="60" cy="18" r="6" fill="#28c840"/>',
            f'<text x="{width / 2:.0f}" y="23" text-anchor="middle" fill="#8b8b8b">'
            "triagewright demo (offline, scripted model)</text>",
            f'<clipPath id="view"><rect x="0" y="0" width="{width:.0f}" height="{view_h:.0f}"/>'
            "</clipPath>",
            f'<g transform="translate(0 {TOP})" clip-path="url(#view)">',
            '<g id="scroll">',
            *body,
            f'<rect id="cover" x="0" y="0" width="{width:.0f}" height="{content + view_h:.0f}" '
            f'fill="{background}"/>',
            "</g>",
            '<g id="summary">',
            *summary,
            "</g>",
            "</g>",
            "</svg>",
            "",
        ]
    )


def main() -> int:
    svg = render(record())
    for private in ("/tmp/", "/home/", "/Users/"):  # noqa: S108 (a check, not a temp file)
        if private in svg:
            raise SystemExit(f"demo.svg would contain a local path ({private})")
    path = tour.ASSETS / "demo.svg"
    path.write_text(svg, encoding="utf-8")
    print(f"wrote docs/assets/demo.svg ({len(svg) // 1024} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
