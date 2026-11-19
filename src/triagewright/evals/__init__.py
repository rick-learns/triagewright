"""Lane F: eval harness and graders. Stub until the contracts land."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args[:1] != ["smoke"]:
        print("usage: triagewright-evals smoke", file=sys.stderr)
        return 2
    print("no eval cases yet")
    return 0
