"""Rendering an :class:`EvalReport` as the markdown report."""

from __future__ import annotations

from triagewright.evals.report.model import EvalReport
from triagewright.evals.report.targets import _pct

_MODE_NOTES: dict[str, str] = {
    "live": "live: latency is wall time per ticket",
    "replay": "replay: latency is estimated (non-model wall time plus recorded model latency)",
    "stub": "stub: no agent or model calls; latency is harness wall time and not gated",
}


def _usd(value: float, sampled: bool) -> str:
    return f"${value:.6f}" if sampled else "n/a (no responses)"


def _ms(value: float, sampled: bool) -> str:
    return f"{value:.0f} ms" if sampled else "n/a (no responses)"


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _met(met: bool | None) -> str:
    return {True: "met", False: "**not met**", None: "not measured"}[met]


def render_markdown(report: EvalReport) -> str:
    s, j = report.summary, report.judge
    models = ", ".join(f"{k}={v}" for k, v in sorted(s.models.items())) or "none"
    estimated = " (estimated)" if s.mode == "replay" else ""
    sha_note = " (the checkout at render time; results don't record it)"
    sha_note = sha_note if report.git_sha_source == "render" else ""
    sampled = any(r.error is None for r in s.results)  # cost and latency need a response
    lines = [
        "# Eval report",
        "",
        f"- Run: `{s.run_id}` at {s.created_at.isoformat()}",
        f"- Git sha: `{s.git_sha}`{sha_note}",
        f"- Mode: {_MODE_NOTES[s.mode]}",
        f"- Provider: {s.provider}",
        f"- Models: {models}",
        "- Live and replay numbers are never compared; the baseline is per mode.",
    ]
    if report.rendered_from:
        lines.append(f"- Rendered from `{report.rendered_from}`, not a fresh run.")
    if report.pending:
        lines.append(
            f"- Pending: {len(report.pending)} selected case(s) have no result yet: "
            + ", ".join(f"`{c}`" for c in report.pending)
        )
    lines += [
        "",
        "## Targets",
        "",
        f"On the {s.n_cases} case(s) in this {s.mode} run.",
        "",
        "| Target | Measured | Goal | Status |",
        "| --- | --- | --- | --- |",
        *(f"| {t.target} | {t.measured} | {t.goal} | {_met(t.met)} |" for t in report.targets),
    ]
    if s.mode != "live":
        lines += ["", "Latency is judged on live runs only (ADR-001)."]
    if s.mode != "stub" and sampled and s.cost_per_ticket_usd == 0:
        # Results don't count model calls, so a run whose calls all failed also costs $0.
        lines += [
            "",
            "Cost is $0: the models are free, or no model call was billed (results don't record "
            "which), so the cost target is met trivially.",
        ]
    lines += [
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Cases | {s.n_cases} |",
        f"| Pass rate | {_pct(s.pass_rate)} |",
        f"| Unsafe actions | {s.unsafe_actions} |",
        f"| Escalation precision | {_pct(s.escalation_precision)} |",
        f"| Escalation recall | {_pct(s.escalation_recall)} |",
        f"| Cost per ticket (judge excluded) | {_usd(s.cost_per_ticket_usd, sampled)} |",
        f"| p50 latency, {s.mode}{estimated} | {_ms(s.p50_latency_ms, sampled)} |",
        f"| p95 latency, {s.mode}{estimated} | {_ms(s.p95_latency_ms, sampled)} |",
        "",
        "## By category",
        "",
        "| Category | Cases | Pass rate | Escalation precision | Escalation recall |",
        "| --- | --- | --- | --- | --- |",
    ]
    for cat in report.categories:
        lines.append(
            f"| {cat.category} | {cat.n_cases} | {_pct(cat.pass_rate)} "
            f"| {_pct(cat.escalation_precision)} | {_pct(cat.escalation_recall)} |"
        )
    rv = report.review
    lines += [
        "",
        "## Review",
        "",
        f"{rv.reviewed} of {rv.total} selected case(s) are human-reviewed (`reviewed: true`).",
    ]
    if rv.unreviewed:
        lines.append("Unreviewed: " + ", ".join(f"`{c}`" for c in rv.unreviewed))
    lines += ["", "## Judge", ""]
    if j.configured:
        lines.append(
            f"Model {j.model}: {j.judged} judged, {j.after_failure} not judged after a "
            f"deterministic failure. Tokens in/out {j.input_tokens}/{j.output_tokens}, "
            f"provider-reported cost ${j.cost_usd:.6f} (not in cost per ticket)."
        )
    else:
        lines.append(
            f"No judge configured: {j.skipped} rubric case(s) graded on deterministic checks only."
        )
    lines += ["", "## Baseline", ""]
    if report.regressions:
        lines += ["**Regression.**", "", *(f"- {r}" for r in report.regressions)]
    else:
        lines.append("No regression.")
    if not report.baseline_compared:
        lines.append(f"No baseline for mode {s.mode}: only unsafe actions were gated.")
    elif report.filtered:
        lines.append("Filtered run: only unsafe actions were gated.")
    lines += ["", "## Failures", ""]
    if not report.failures:
        lines.append("None.")
    else:
        lines += [
            "| Case | Category | Check | Expected | Actual |",
            "| --- | --- | --- | --- | --- |",
        ]
    for f in report.failures:
        for c in f.checks:
            lines.append(
                f"| `{f.case_id}` | {f.category} | {c.grader} | {_cell(c.expected)} "
                f"| {_cell(c.actual)} |"
            )
    return "\n".join(lines) + "\n"
