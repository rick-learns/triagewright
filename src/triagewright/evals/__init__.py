"""Lane F: eval harness and graders.

Until lane F lands, ``triagewright-evals smoke`` only validates every case file
against the :class:`~triagewright.contracts.evals.EvalCase` contract.
"""

from __future__ import annotations

import sys
from pathlib import Path

from triagewright.data import load_eval_cases


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args[:1] != ["smoke"]:
        print("usage: triagewright-evals smoke [CASES_DIR]", file=sys.stderr)
        return 2
    cases_dir = Path(args[1]) if len(args) > 1 else Path("evals/cases")
    cases = load_eval_cases(cases_dir)
    print(f"validated {len(cases)} eval cases in {cases_dir} (harness not implemented yet)")
    return 0
