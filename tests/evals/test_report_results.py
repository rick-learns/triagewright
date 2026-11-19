"""Reading a results JSONL, including a torn final line."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.evals.test_report_support import rec
from triagewright.evals.report import ResultsError, read_results
from triagewright.evals.runner import CaseRecord


def write_lines(path: Path, *records: CaseRecord, tail: str = "") -> None:
    path.write_text("".join(r.model_dump_json() + "\n" for r in records) + tail)


def test_read_results_skips_a_torn_final_line(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    write_lines(path, rec("happy-001"), rec("happy-002"), tail='{"key": "happy-0')
    records, warnings = read_results(path)
    assert [r.result.case_id for r in records] == ["happy-001", "happy-002"]
    assert warnings == [f"{path}:3: skipped a torn final line"]

    write_lines(path, rec("happy-001"), tail="\n\n")
    assert read_results(path) == ([rec("happy-001")], [])

    whole = rec("happy-002").model_dump_json().replace("happy-002", "caf\u00e9")
    torn = whole.encode()[: whole.encode().index(b"\xa9")]  # cut inside the UTF-8 for e-acute
    path.write_bytes(rec("happy-001").model_dump_json().encode() + b"\n" + torn)
    assert read_results(path) == ([rec("happy-001")], [f"{path}:2: skipped a torn final line"])

    path.write_text(rec("happy-001").model_dump_json())  # complete but unterminated: kept
    assert read_results(path) == ([rec("happy-001")], [])


def test_read_results_rejects_a_bad_line_before_the_end(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    path.write_text('{"torn\n' + rec("happy-001").model_dump_json() + "\n")
    with pytest.raises(ResultsError, match=r"results\.jsonl:1"):
        read_results(path)
    path.write_text(rec("happy-001").model_dump_json() + '\n{"key": 1}\n')  # terminated: corrupt
    with pytest.raises(ResultsError, match=r"results\.jsonl:2"):
        read_results(path)
