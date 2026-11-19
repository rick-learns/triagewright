"""Lane F: eval harness, graders, and report.

``triagewright-evals smoke`` validates every case file against the
:class:`~triagewright.contracts.evals.EvalCase` contract. ``triagewright-evals
run`` resolves every case on its own isolated stack, grades the evidence,
writes the report, and exits non-zero on a regression against
``evals/baseline.json``. See ``evals/README.md``.
"""

from __future__ import annotations

from triagewright.evals.cli import main

__all__ = ["main"]
