"""Run aggregation: percentiles, the summary math, and judge usage."""

from __future__ import annotations

import pytest

from tests.evals.test_report_support import NOW, rec, summary_of
from triagewright.contracts.evals import EvalCategory
from triagewright.contracts.llm import Usage
from triagewright.evals.report import judge_report, percentile


@pytest.mark.parametrize(
    ("values", "q", "want"),
    [
        ([], 95, 0.0),
        ([7.0], 95, 7.0),
        ([1.0, 2.0, 3.0, 4.0], 50, 2.5),
        ([4.0, 1.0, 3.0, 2.0], 95, 3.85),  # unsorted input; 1 + 0.95 * 3 = 3.85 -> 3.85
        ([float(i) for i in range(1, 21)], 95, 19.05),
        ([10.0, 20.0], 0, 10.0),
        ([10.0, 20.0], 100, 20.0),
    ],
)
def test_percentile_interpolates_linearly(values: list[float], q: float, want: float) -> None:
    assert percentile(values, q) == pytest.approx(want)


def test_summary_math() -> None:
    s = summary_of(
        # TP, FP, FN, TN: precision 1/2, recall 1/2
        rec("escal-001", category=EvalCategory.ESCALATION_REQUIRED, expected=True, escalated=True),
        rec("escal-002", category=EvalCategory.ESCALATION_REQUIRED, passed=False, escalated=True),
        rec("happy-001", passed=False, expected=True, latency=300.0, cost=0.003),
        rec("happy-002", latency=200.0, cost=0.001),
        rec("happy-003", passed=False, error="timed out", latency=60_000.0, cost=9.0),
    )
    assert s.n_cases == 5
    assert s.pass_rate == pytest.approx(2 / 5)
    assert s.escalation_precision == pytest.approx(0.5)
    assert s.escalation_recall == pytest.approx(0.5)
    assert s.by_category == {"escalation_required": 0.5, "happy_path": pytest.approx(1 / 3)}
    # The errored case is excluded from cost and latency: [100, 100, 300, 200].
    assert s.cost_per_ticket_usd == pytest.approx(0.004 / 4)
    assert s.p50_latency_ms == pytest.approx(150.0)
    assert s.p95_latency_ms == pytest.approx(285.0)
    assert (s.mode, s.provider, s.git_sha, s.created_at) == ("replay", "replay", "abc123", NOW)
    assert s.run_id.startswith("run_") and len(s.results) == 5


def test_summary_unsafe_and_undefined_rates() -> None:
    s = summary_of(rec("inject-001", unsafe=True, passed=False), rec("inject-002"))
    assert s.unsafe_actions == 1
    assert s.escalation_precision is None and s.escalation_recall is None
    empty = summary_of()
    assert (empty.n_cases, empty.pass_rate, empty.p95_latency_ms) == (0, 0.0, 0.0)


def test_judge_report_counts_and_tokens() -> None:
    usage = Usage(input_tokens=10, output_tokens=2)
    records = [
        rec("happy-001", judge_status="judged", judge_usage=usage),
        rec("happy-002", judge_status="judged", judge_usage=usage),
        rec("happy-003", judge_status="after_failure"),
        rec("happy-004"),
    ]
    j = judge_report(records, model="gemini-3.8-flash")
    assert (j.configured, j.judged, j.after_failure, j.skipped) == (True, 2, 1, 0)
    assert (j.input_tokens, j.output_tokens) == (20, 4)
    skipped = judge_report([rec("happy-001", judge_status="skipped")], model=None)
    assert not skipped.configured and skipped.skipped == 1
