from __future__ import annotations

import argparse
import itertools
import json
import subprocess
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]

JOB_DIR = ROOT / "experiments" / "jobs"
LOG_DIR = ROOT / "logs"

JOB_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate and optionally submit ARC SLURM sweep."
    )

    parser.add_argument("--shots", type=int, required=True)

    parser.add_argument("--distances",
                        type=int,
                        nargs="+",
                        required=True)

    parser.add_argument("--beams",
                        type=int,
                        nargs="+",
                        required=True)

    parser.add_argument("--p-values",
                        type=float,
                        nargs="+",
                        required=True)

    parser.add_argument("--pqlimits",
                        type=int,
                        nargs="+",
                        required=True)

    parser.add_argument("--threads",
                        type=int,
                        default=32)

    parser.add_argument("--workers",
                        type=int,
                        default=1)

    parser.add_argument("--parallel",
                        type=int,
                        default=12)

    parser.add_argument("--partition",
                        default="medium")

    parser.add_argument("--time",
                        default="24:00:00")

    parser.add_argument("--memory",
                        default="16G")

    parser.add_argument("--run-group",
                        required=True)

    parser.add_argument("--submit",
                        action="store_true")

    return parser.parse_args()

def generate_combinations(args):
    combos = list(itertools.product(
        args.distances,
        args.beams,
        args.p_values,
        args.pqlimits,
    ))
    return combos


def main():

    args = parse_args()
    combos = generate_combinations(args)
    print(f"{len(combos)} parameter combinations\n")

    for i, (d, beam, p, pq) in enumerate(combos):
        print(
            f"{i:2d}: "
            f"d={d:<2} "
            f"beam={beam:<3} "
            f"p={p:<6} "
            f"pq={pq}"
        )


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
