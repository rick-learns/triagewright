"""The per-mode baseline gate and the baseline file."""

from __future__ import annotations

from pathlib import Path

from tests.evals.test_report_support import NOW, rec, summary_of
from triagewright.contracts.evals import EvalRunSummary
from triagewright.evals.report import (
    Baseline,
    BaselineEntry,
    load_baseline,
    regressions,
    updated_baseline,
)
from triagewright.testing import REPO_ROOT


def entry(pass_rate: float = 0.8, cost: float = 0.01, p95: float = 1000.0) -> BaselineEntry:
    return BaselineEntry(
        pass_rate=pass_rate,
        cost_per_ticket_usd=cost,
        p95_latency_ms=p95,
        n_cases=10,
        git_sha="old",
        provider="replay",
        models={},
        created_at=NOW,
    )


def with_metrics(pass_rate: float, cost: float, p95: float, *, unsafe: int = 0) -> EvalRunSummary:
    s = summary_of(rec("happy-001"))
    update = {"pass_rate": pass_rate, "cost_per_ticket_usd": cost, "p95_latency_ms": p95}
    return s.model_copy(update={**update, "unsafe_actions": unsafe})


def test_no_regression_within_tolerance() -> None:
    base = Baseline(modes={"replay": entry()})
    assert regressions(with_metrics(0.8, 0.011, 1100.0), base) == []  # exactly +10% is allowed
    assert regressions(with_metrics(0.9, 0.0, 10.0), base) == []


def test_each_metric_regresses() -> None:
    base = Baseline(modes={"replay": entry()})
    found = regressions(with_metrics(0.7, 0.0111, 1101.0, unsafe=2), base)
    assert len(found) == 4
    assert found[0] == "2 unsafe action(s)"
    assert found[1].startswith("pass rate dropped: 70.0% < 80.0%")
    assert found[2].startswith("cost per ticket rose more than 10%")
    assert found[3].startswith("p95 latency rose more than 10%")


def test_free_baseline_cost_regresses_on_any_spend() -> None:
    base = Baseline(modes={"replay": entry(cost=0.0)})
    assert regressions(with_metrics(0.8, 0.0, 1000.0), base) == []
    assert len(regressions(with_metrics(0.8, 0.0001, 1000.0), base)) == 1


def test_baseline_is_per_mode() -> None:
    base = Baseline(modes={"live": entry(pass_rate=1.0, p95=1.0)})
    assert regressions(with_metrics(0.1, 1.0, 9e9), base) == []  # no replay baseline
    assert regressions(with_metrics(0.1, 1.0, 9e9, unsafe=1), base) == ["1 unsafe action(s)"]


def test_filtered_runs_gate_only_unsafe_actions() -> None:
    base = Baseline(modes={"replay": entry()})
    assert regressions(with_metrics(0.0, 1.0, 9e9), base, compare_metrics=False) == []
    found = regressions(with_metrics(0.0, 1.0, 9e9, unsafe=1), base, compare_metrics=False)
    assert found == ["1 unsafe action(s)"]


def test_stub_latency_is_not_gated() -> None:
    base = Baseline(modes={"stub": entry(p95=0.01)})
    stub = summary_of(rec("happy-001", latency=5.0), mode="stub")
    assert regressions(stub, base) == []


def test_baseline_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    assert load_baseline(path) == Baseline()
    s = summary_of(rec("happy-001", latency=50.0, cost=0.002))
    new = updated_baseline(Baseline(modes={"live": entry()}), s)
    path.write_text(new.model_dump_json(indent=2))
    loaded = load_baseline(path)
    assert set(loaded.modes) == {"live", "replay"}
    assert loaded.modes["replay"].p95_latency_ms == 50.0
    assert loaded.modes["replay"].cost_per_ticket_usd == 0.002
    assert regressions(s, loaded) == []


def test_committed_baseline_is_valid() -> None:
    load_baseline(REPO_ROOT / "evals" / "baseline.json")
