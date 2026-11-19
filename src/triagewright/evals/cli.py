"""``triagewright-evals``: ``smoke`` validates every case file; ``run`` is the release gate."""

from __future__ import annotations

import argparse
import asyncio
import fnmatch
import importlib
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from triagewright.config import Settings, get_settings
from triagewright.contracts.common import canonical_json
from triagewright.contracts.llm import Provider
from triagewright.contracts.ports import CaseRuntimeFactory
from triagewright.data import load_as_of, load_eval_cases, load_world
from triagewright.evals.judge import JudgeClientFactory
from triagewright.evals.report import (
    EvalReport,
    judge_report,
    load_baseline,
    regressions,
    render_markdown,
    summarize,
    updated_baseline,
)
from triagewright.evals.runner import JudgeConfig, Mode, RunConfig, run_cases

DEFAULT_FACTORY = "triagewright.wiring:case_runtime_factory"


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
    run.add_argument("--timeout", type=float, default=60.0, help="per-case timeout in seconds")
    stack = run.add_mutually_exclusive_group()
    stack.add_argument(
        "--factory",
        default=DEFAULT_FACTORY,
        metavar="MODULE:ATTR",
        help="zero-argument callable returning a CaseRuntimeFactory",
    )
    stack.add_argument(
        "--stub", action="store_true", help="fake stack with the stub resolver (no agent)"
    )
    run.add_argument(
        "--judge",
        metavar="MODULE:ATTR",
        help="zero-argument callable returning a JudgeClientFactory; unset skips the judge",
    )
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


def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
    all_cases = load_eval_cases(args.cases)
    cases = all_cases
    if args.case:
        cases = [c for c in cases if any(fnmatch.fnmatchcase(c.id, g) for g in args.case)]
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
    if judge_factory is not None and not callable(judge_factory):
        print(f"{args.judge} did not return a JudgeClientFactory", file=sys.stderr)
        return 2

    mode = _mode(settings, args.stub)
    defaults = settings.models()
    if mode == "stub":
        provider, models = "none (stub resolver, no model calls)", {}
    else:
        provider = settings.llm_provider.value
        models = {k: v for k, v in defaults.items() if k != "judge"}
    judge = None
    if judge_factory is not None:
        judge = JudgeConfig(
            cast(JudgeClientFactory, judge_factory), defaults["judge"], settings.recordings_dir
        )
        models["judge"] = judge.model
    config = RunConfig(
        mode=mode,
        concurrency=args.concurrency,
        min_interval_s=args.min_interval,
        case_timeout_s=args.timeout,
        results_path=args.results,
        resume=args.resume,
        stack=canonical_json({"provider": provider, "models": models}),
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
    report = EvalReport(
        summary=summary,
        judge=judge_report(records, model=judge.model if judge else None),
        baseline_compared=mode in baseline.modes,
        filtered=filtered,
        regressions=regressions(summary, baseline, compare_metrics=not filtered),
    )
    for path, text in (
        (args.report, render_markdown(report)),
        (args.json, report.model_dump_json(indent=2) + "\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

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
    parser.print_usage(sys.stderr)
    return 2
