"""Reading a results JSONL back into case records, tolerating a torn final line."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from triagewright.evals.runner import CaseRecord


class ResultsError(ValueError):
    """The results JSONL can't be read (a bad line before the last one)."""


def read_results(path: Path) -> tuple[list[CaseRecord], list[str]]:
    """Every record in a results JSONL, in file order, plus warnings.

    An interrupted run can leave a torn final line; it's skipped with a warning.
    A bad line anywhere else raises :class:`ResultsError`.
    """
    *lines, tail = path.read_bytes().split(b"\n")  # tail: text after the last newline
    records: list[CaseRecord] = []
    warnings: list[str] = []
    for n, line in enumerate([*lines, tail], 1):
        if not line.strip():
            continue
        try:
            records.append(CaseRecord.model_validate_json(line))
        except ValidationError as exc:  # also covers bad JSON and bad UTF-8
            if n <= len(lines):  # a complete line: the file is corrupt, not torn
                raise ResultsError(f"{path}:{n}: not a case record: {exc}") from exc
            warnings.append(f"{path}:{n}: skipped a torn final line")
    return records, warnings
