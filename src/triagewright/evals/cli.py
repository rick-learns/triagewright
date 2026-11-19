"""``triagewright-evals``: ``smoke`` validates every case file; ``run`` is the release gate.

``report`` re-renders the reports from a results JSONL, and ``review`` lists unreviewed cases.
"""

from __future__ import annotations

import argparse
import asyncio
import fnmatch
import importlib
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from triagewright.config import Settings, get_settings
from triagewright.contracts.common import canonical_json
from triagewright.contracts.evals import EvalCase
from triagewright.contracts.llm import Provider
from triagewright.contracts.ports import CaseRuntimeFactory
from triagewright.data import load_as_of, load_eval_cases, load_world
from triagewright.evals.judge import JudgeClientFactory
from triagewright.evals.report import (
    EvalReport,
    ResultsError,
    build_report,
    judge_report,
    load_baseline,
    read_results,
    render_markdown,
    review_status,
    summarize,
    updated_baseline,
)
from triagewright.evals.runner import JudgeConfig, Mode, RunConfig, input_key, run_cases

DEFAULT_FACTORY = "triagewright.wiring:case_runtime_factory"


def _add_stack_args(parser: argparse.ArgumentParser) -> None:
    """The flags that identify a run's stack; they are part of every result's resume key."""
    parser.add_argument("--timeout", type=float, default=60.0, help="per-case timeout in seconds")
    stack = parser.add_mutually_exclusive_group()
    stack.add_argument(
        "--factory",
        default=DEFAULT_FACTORY,
        metavar="MODULE:ATTR",
        help="zero-argument callable returning a CaseRuntimeFactory",
    )
    stack.add_argument(
        "--stub", action="store_true", help="fake stack with the stub resolver (no agent)"
    )
    parser.add_argument(
        "--judge",
        metavar="MODULE:ATTR",
        help="zero-argument callable returning a JudgeClientFactory; unset skips the judge",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="triagewright-evals")
    sub = parser.add_subparsers(dest="command")

    smoke = sub.add_parser("smoke", help="validate every case file against the EvalCase contract")
    smoke.add_argument("cases_dir", nargs="?", type=Path, default=Path("evals/cases"))

    run = sub.add_parser("run", help="run the cases, write the report, and gate on the baseline")
    run.add_argument("--cases", type=Path, default=Path("evals/cases"), help="cases directory")
    run.add_argument(
        "--case", action="append", default=[], metavar="GLOB", help="only ids matching (repeat)"
    )
    run.add_argument("--report", type=Path, default=Path("var/evals/report.md"))
    run.add_argument("--json", type=Path, default=Path("var/evals/report.json"))
    run.add_argument("--results", type=Path, default=Path("var/evals/results.jsonl"))
    run.add_argument("--resume", action="store_true", help="skip cases already in --results")
    run.add_argument("--baseline", type=Path, default=Path("evals/baseline.json"))
    run.add_argument(
        "--update-baseline", action="store_true", help="store this run as the mode's baseline"
    )
    run.add_argument("--concurrency", type=int, default=4)
    run.add_argument("--min-interval", type=float, default=0.0, help="seconds between case starts")
    _add_stack_args(run)

    report = sub.add_parser(
        "report", help="re-render the reports from a results JSONL without running anything"
    )
    report.add_argument("--cases", type=Path, default=Path("evals/cases"), help="cases directory")
    report.add_argument(
        "--case", action="append", default=[], metavar="GLOB", help="only ids matching (repeat)"
    )
    report.add_argument("--results", type=Path, default=Path("var/evals/results.jsonl"))
    report.add_argument("--report", type=Path, default=Path("var/evals/report.md"))
    report.add_argument("--json", type=Path, default=Path("var/evals/report.json"))
    report.add_argument("--baseline", type=Path, default=Path("evals/baseline.json"))
    _add_stack_args(report)  # pass the run's flags; only results from that run count

    review = sub.add_parser("review", help="list the cases not yet human-reviewed, by category")
    review.add_argument("cases_dir", nargs="?", type=Path, default=Path("evals/cases"))
    return parser


def _load(spec: str) -> object:
    module, _, attr = spec.partition(":")
    builder = getattr(importlib.import_module(module), attr)
    return builder()


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def _mode(settings: Settings, stub: bool) -> Mode:
    if stub:
        return "stub"
    return "replay" if settings.llm_provider is Provider.REPLAY else "live"


def _smoke(cases_dir: Path) -> int:
    cases = load_eval_cases(cases_dir)
    print(f"validated {len(cases)} eval cases in {cases_dir}")
    return 0


def _select(cases: list[EvalCase], globs: list[str]) -> list[EvalCase]:
    if not globs:
        return cases
    return [c for c in cases if any(fnmatch.fnmatchcase(c.id, g) for g in globs)]


@dataclass(frozen=True)
class _Identity:
    """What a run is, from settings and flags. ``stack`` is part of each result's key."""

    mode: Mode
    provider: str
    models: dict[str, str]  # with the judge model when a judge is set
    judge_model: str | None
    stack: str


def _identity(settings: Settings, args: argparse.Namespace) -> _Identity:
    mode = _mode(settings, args.stub)
    defaults = settings.models()
    if mode == "stub":
        provider, models = "none (stub resolver, no model calls)", {}
    else:
        provider = settings.llm_provider.value
        models = {k: v for k, v in defaults.items() if k != "judge"}
    judge_model = defaults["judge"] if args.judge else None
    if judge_model is not None:
        models["judge"] = judge_model
    stack = canonical_json(
        {
            "provider": provider,
            "models": models,
            "factory": "stub" if args.stub else args.factory,
            "judge": args.judge,
        }
    )
    return _Identity(mode, provider, models, judge_model, stack)


def _write(report: EvalReport, md_path: Path, json_path: Path) -> None:
    for path, text in (
        (md_path, render_markdown(report)),
        (json_path, report.model_dump_json(indent=2) + "\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
    all_cases = load_eval_cases(args.cases)
    cases = _select(all_cases, args.case)
    if not cases:
        print(f"no eval cases selected in {args.cases}", file=sys.stderr)
        return 2
    try:
        if args.stub:
            from triagewright.testing import FakeCaseRuntimeFactory

            factory: object = FakeCaseRuntimeFactory()
        else:
            factory = _load(args.factory)
        judge_factory = _load(args.judge) if args.judge else None
    except (ImportError, AttributeError, ValueError) as exc:
        print(f"cannot load the eval stack: {exc!r} (--stub runs without one)", file=sys.stderr)
        return 2
    if not isinstance(factory, CaseRuntimeFactory):
        print(f"{args.factory} did not return a CaseRuntimeFactory", file=sys.stderr)
        return 2
    if args.judge and not callable(judge_factory):
        print(f"{args.judge} did not return a JudgeClientFactory", file=sys.stderr)
        return 2

    run = _identity(settings, args)
    mode, provider, models = run.mode, run.provider, run.models
    judge = None
    if judge_factory is not None and run.judge_model is not None:
        judge = JudgeConfig(
            cast(JudgeClientFactory, judge_factory), run.judge_model, settings.recordings_dir
        )
    config = RunConfig(
        mode=mode,
        concurrency=args.concurrency,
        min_interval_s=args.min_interval,
        case_timeout_s=args.timeout,
        results_path=args.results,
        resume=args.resume,
        stack=run.stack,
    )
    records = asyncio.run(
        run_cases(
            factory,
            cases,
            seed_world=load_world(settings.seed_dir),
            seed_as_of=load_as_of(settings.seed_dir),
            config=config,
            judge=judge,
        )
    )

    summary = summarize(
        records,
        mode=mode,
        provider=provider,
        models=models,
        git_sha=_git_sha(),
        created_at=datetime.now(UTC),
    )
    baseline = load_baseline(args.baseline)
    filtered = len(cases) < len(all_cases)
    report = build_report(
        summary,
        cases=cases,
        judge=judge_report(records, model=judge.model if judge else None),
        baseline=baseline,
        filtered=filtered,
    )
    _write(report, args.report, args.json)

    print(
        f"{summary.n_cases} cases ({mode}): pass rate {summary.pass_rate:.1%}, "
        f"{summary.unsafe_actions} unsafe; report {args.report}, json {args.json}"
    )
    for reason in report.regressions:
        print(f"regression: {reason}", file=sys.stderr)
    if args.update_baseline:
        if filtered or summary.unsafe_actions:
            print("baseline not updated: filtered run or unsafe actions", file=sys.stderr)
        else:
            new = updated_baseline(baseline, summary)
            args.baseline.write_text(new.model_dump_json(indent=2) + "\n", encoding="utf-8")
            print(f"baseline for {mode} written to {args.baseline}")
    return 1 if report.regressions else 0


def _report(args: argparse.Namespace) -> int:
    """Render the reports from ``--results`` as they stand, e.g. mid-way through a resumed run."""
    settings = get_settings()
    all_cases = load_eval_cases(args.cases)
    cases = _select(all_cases, args.case)
    if not cases:
        print(f"no eval cases selected in {args.cases}", file=sys.stderr)
        return 2
    if not args.results.exists():
        print(f"no results file at {args.results}", file=sys.stderr)
        return 2
    try:
        saved, warnings = read_results(args.results)
    except ResultsError as exc:
        print(f"cannot read the results: {exc}", file=sys.stderr)
        return 2
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    # Like --resume, a record counts only if its key matches the case and these run flags, so
    # edited cases and other stacks' results are pending, and provider and models are the run's.
    run = _identity(settings, args)
    by_key = {r.key: r for r in saved}
    records, pending = [], []
    for case in cases:
        key = input_key(
            case,
            mode=run.mode,
            timeout_s=args.timeout,
            stack=run.stack,
            judge_model=run.judge_model,
        )
        if key in by_key:
            records.append(by_key[key])
        else:
            pending.append(case.id)
    if not records:
        print(
            f"no results in {args.results} match the selected cases and run flags",
            file=sys.stderr,
        )
        return 2

    mode = run.mode
    summary = summarize(
        records,
        mode=mode,
        provider=run.provider,
        models=run.models,
        git_sha=_git_sha(),
        created_at=datetime.now(UTC),
    )
    report = build_report(
        summary,
        cases=cases,
        judge=judge_report(records, model=run.judge_model),
        baseline=load_baseline(args.baseline),
        filtered=len(records) < len(all_cases),
        pending=pending,
        rendered_from=str(args.results),
    )
    _write(report, args.report, args.json)
    print(
        f"{summary.n_cases} cases ({mode}), {len(pending)} pending: pass rate "
        f"{summary.pass_rate:.1%}, {summary.unsafe_actions} unsafe; "
        f"report {args.report}, json {args.json}"
    )
    for reason in report.regressions:
        print(f"regression: {reason}", file=sys.stderr)
    return 1 if report.regressions else 0


def _review(cases_dir: Path) -> int:
    """List unreviewed cases by category. Marking a case reviewed stays a human edit."""
    cases = load_eval_cases(cases_dir)
    status = review_status(cases)
    print(f"{status.reviewed} of {status.total} cases reviewed; checklist: evals/REVIEWING.md")
    by_category: dict[str, list[str]] = {}
    for case in cases:
        if not case.reviewed:
            by_category.setdefault(case.category.value, []).append(case.id)
    for category, ids in sorted(by_category.items()):
        print(f"{category} ({len(ids)}): {', '.join(ids)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    try:
        args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    except SystemExit as exc:  # argparse exits on --help and on usage errors
        return exc.code if isinstance(exc.code, int) else 2
    if args.command == "smoke":
        return _smoke(args.cases_dir)
    if args.command == "run":
        return _run(args)
    if args.command == "report":
        return _report(args)
    if args.command == "review":
        return _review(args.cases_dir)
    parser.print_usage(sys.stderr)
    return 2
