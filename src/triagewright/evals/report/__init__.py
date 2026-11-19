"""Aggregation into :class:`EvalRunSummary`, the baseline gate, and the markdown/JSON report.

Cost per ticket and latency cover cases that produced a response (no
``error``); errored cases count against the pass rate instead. Judge calls are
never part of cost per ticket; the report lists them separately.

Modules: ``aggregate`` (summary, category stats, judge usage), ``baseline`` (the
per-mode gate), ``results`` (reading a results JSONL), ``targets`` (release
targets and review status), ``failures`` (failing checks with their
expectations), ``model`` (:class:`EvalReport` and :func:`build_report`), and
``markdown`` (rendering). Every name the old ``report.py`` defined stays importable here.
"""

from __future__ import annotations

from triagewright.evals.report.aggregate import CategoryStats as CategoryStats
from triagewright.evals.report.aggregate import JudgeReport as JudgeReport
from triagewright.evals.report.aggregate import _ratio as _ratio
from triagewright.evals.report.aggregate import category_stats as category_stats
from triagewright.evals.report.aggregate import escalation_rates as escalation_rates
from triagewright.evals.report.aggregate import judge_report as judge_report
from triagewright.evals.report.aggregate import percentile as percentile
from triagewright.evals.report.aggregate import summarize as summarize
from triagewright.evals.report.baseline import _EPS as _EPS
from triagewright.evals.report.baseline import COST_TOLERANCE as COST_TOLERANCE
from triagewright.evals.report.baseline import LATENCY_TOLERANCE as LATENCY_TOLERANCE
from triagewright.evals.report.baseline import Baseline as Baseline
from triagewright.evals.report.baseline import BaselineEntry as BaselineEntry
from triagewright.evals.report.baseline import load_baseline as load_baseline
from triagewright.evals.report.baseline import regressions as regressions
from triagewright.evals.report.baseline import updated_baseline as updated_baseline
from triagewright.evals.report.failures import FailedCheck as FailedCheck
from triagewright.evals.report.failures import FailureDetail as FailureDetail
from triagewright.evals.report.failures import _failed_check as _failed_check
from triagewright.evals.report.failures import _list as _list
from triagewright.evals.report.failures import expected_for as expected_for
from triagewright.evals.report.failures import failure_details as failure_details
from triagewright.evals.report.markdown import _MODE_NOTES as _MODE_NOTES
from triagewright.evals.report.markdown import _cell as _cell
from triagewright.evals.report.markdown import _met as _met
from triagewright.evals.report.markdown import _ms as _ms
from triagewright.evals.report.markdown import _usd as _usd
from triagewright.evals.report.markdown import render_markdown as render_markdown
from triagewright.evals.report.model import EvalReport as EvalReport
from triagewright.evals.report.model import build_report as build_report
from triagewright.evals.report.results import ResultsError as ResultsError
from triagewright.evals.report.results import read_results as read_results
from triagewright.evals.report.targets import TARGET_COST_USD as TARGET_COST_USD
from triagewright.evals.report.targets import TARGET_P95_MS as TARGET_P95_MS
from triagewright.evals.report.targets import TARGET_PASS_RATE as TARGET_PASS_RATE
from triagewright.evals.report.targets import TARGET_REVIEWED as TARGET_REVIEWED
from triagewright.evals.report.targets import ReviewStatus as ReviewStatus
from triagewright.evals.report.targets import TargetCheck as TargetCheck
from triagewright.evals.report.targets import _pct as _pct
from triagewright.evals.report.targets import review_status as review_status
from triagewright.evals.report.targets import target_checks as target_checks
