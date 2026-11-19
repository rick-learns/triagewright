"""``python -m triagewright.datagen --seed 7 --customers 20 --tickets 30 --out DIR``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from triagewright.datagen.generator import generate, parse_mix, write


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m triagewright.datagen",
        description="Write a synthetic world, tickets, and their intended outcomes.",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--customers", type=int, required=True)
    parser.add_argument("--tickets", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--mix",
        help="category weights, e.g. refund_request=3,billing=1 (default: refunds 3, others 1)",
    )
    args = parser.parse_args(argv)
    try:
        mix = parse_mix(args.mix) if args.mix else None
        gen = generate(args.seed, args.customers, args.tickets, mix)
    except ValueError as exc:
        parser.error(str(exc))
    for path in write(gen, args.out):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
