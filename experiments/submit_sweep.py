from __future__ import annotations

import argparse
import itertools
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

    parser.add_argument(
        "--basis",
        default="surface_code_X",
    )

    parser.add_argument(
        "--distances",
        type=int,
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--beams",
        type=int,
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--p-values",
        type=float,
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--pqlimits",
        type=int,
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--stim-dir",
        default="testdata/surfacecodes",
        help="Directory containing the Stim benchmark circuits.",
    )

    parser.add_argument(
        "--beam-climbing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable beam climbing in generated jobs.",
    )

    parser.add_argument(
        "--merge-errors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable error merging in generated jobs.",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--parallel",
        type=int,
        default=None,
        help="Deprecated alias for --array-percent; use --array-percent instead.",
    )

    parser.add_argument(
        "--array-percent",
        type=int,
        default=16,
        help="Maximum percentage of simultaneously running array tasks.",
    )

    parser.add_argument(
        "--partition",
        default="medium",
    )

    parser.add_argument(
        "--time",
        default="24:00:00",
    )

    parser.add_argument(
        "--memory",
        default="16G",
    )

    parser.add_argument(
        "--run-group",
        required=True,
    )

    parser.add_argument(
        "--submit",
        action="store_true",
    )

    return parser.parse_args()


def generate_combinations(args):
    combos = list(
        itertools.product(
            args.distances,
            args.beams,
            args.p_values,
            args.pqlimits,
        )
    )
    return combos


def write_slurm_script(args, combos):
    script = JOB_DIR / f"{args.run_group}.sh"

    n = len(combos)

    distances = " ".join(str(c[0]) for c in combos)
    beams = " ".join(str(c[1]) for c in combos)
    pvalues = " ".join(str(c[2]) for c in combos)
    pqlimits = " ".join(str(c[3]) for c in combos)

    beam_climbing_flag = (
        "--beam-climbing"
        if args.beam_climbing
        else "--no-beam-climbing"
    )

    merge_errors_flag = (
        "--merge-errors"
        if args.merge_errors
        else "--no-merge-errors"
    )

    text = dedent(
        f"""\
        #!/bin/bash
        #SBATCH --job-name={args.run_group}
        #SBATCH --partition={args.partition}
        #SBATCH --time={args.time}
        #SBATCH --cpus-per-task={args.threads}
        #SBATCH --mem={args.memory}
        #SBATCH --array=0-{n-1}%{args.array_percent}

        #SBATCH --output={LOG_DIR}/{args.run_group}_%A_%a.out
        #SBATCH --error={LOG_DIR}/{args.run_group}_%A_%a.err

        set -euo pipefail

        cd "{ROOT}"

        module purge
        module load Python/3.13.5-GCCcore-14.3.0
        module load Bazel/7.7.0-GCCcore-14.3.0-Java-21

        source .venv/bin/activate

        DISTANCES=({distances})
        BEAMS=({beams})
        PVALUES=({pvalues})
        PQLIMITS=({pqlimits})

        IDX=$SLURM_ARRAY_TASK_ID

        DIST=${{DISTANCES[$IDX]}}
        BEAM=${{BEAMS[$IDX]}}
        PVALUE=${{PVALUES[$IDX]}}
        PQLIMIT=${{PQLIMITS[$IDX]}}

        echo "======================================================"
        echo "Task $IDX"
        echo "distance = $DIST"
        echo "beam     = $BEAM"
        echo "p        = $PVALUE"
        echo "pqlimit  = $PQLIMIT"
        echo "stim_dir = {args.stim_dir}"
        echo "beam climbing = {args.beam_climbing}"
        echo "merge errors   = {args.merge_errors}"
        echo "======================================================"

        ./bazel-bin/src/py/run_tesseract \\
          --n-shots {args.shots} \\
          --max-files 1 \\
          --workers {args.workers} \\
          --threads {args.threads} \\
          --decode-mode batch \\
          --stim-dir {args.stim_dir} \\
          --basis {args.basis} \\
          --distances $DIST \\
          --p-values $PVALUE \\
          --det-beam $BEAM \\
          {beam_climbing_flag} \\
          {merge_errors_flag} \\
          --pqlimit $PQLIMIT \\
          --run-group {args.run_group}
        """
    )

    script.write_text(text)
    script.chmod(0o755)

    return script


def main():
    args = parse_args()

    if args.parallel is not None:
        args.array_percent = args.parallel

    combos = generate_combinations(args)

    print(f"{len(combos)} jobs")
    script = write_slurm_script(args, combos)

    print()
    print("Generated:")
    print(script)


if __name__ == "__main__":
    main()
