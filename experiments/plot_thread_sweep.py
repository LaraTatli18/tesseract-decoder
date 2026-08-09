from __future__ import annotations

"""Plot decoder runtime and throughput as a function of thread count.

This script compares benchmark runs across different decoder thread settings and
produces line plots for decode time and shots per second.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from plot_utils import (
    collect_plot_runs,
    expand_run_dirs,
    matches_filters,
    save_figure,
    setup_matplotlib,
)


@dataclass(frozen=True)
class ThreadSweepRun:
    threads: int
    decode_time_seconds: float
    shots_per_second: float
    run_dir: Path
    manifest: dict[str, Any]


def _collect_runs(
    run_dirs: list[Path],
    args: argparse.Namespace,
) -> list[ThreadSweepRun]:
    runs: list[ThreadSweepRun] = []

    for run in collect_plot_runs(run_dirs):
        if not matches_filters(run.manifest, args):
            continue

        row = run.rows[0]

        try:
            threads = int(run.manifest["threads"])
            decode_time_seconds = float(row["decode_time_seconds"])
            shots_per_second = float(row["shots_per_second"])
        except (KeyError, TypeError, ValueError):
            print(f"Skipping {run.run_dir}: invalid thread-sweep data")
            continue

        runs.append(
            ThreadSweepRun(
                threads=threads,
                decode_time_seconds=decode_time_seconds,
                shots_per_second=shots_per_second,
                run_dir=run.run_dir,
                manifest=run.manifest,
            )
        )

    runs.sort(key=lambda run: run.threads)
    return runs


def _plot_metric(
    x: list[int],
    y: list[float],
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    output: Path,
    annotate_points: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(x, y, marker="o")

    if annotate_points:
        for xi, yi in zip(x, y):
            ax.annotate(
                str(xi),
                (xi, yi),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure(fig, output)
    plt.close(fig)


def main() -> int:
    setup_matplotlib()

    parser = argparse.ArgumentParser(
        description="Plot results from a decoder thread sweep."
    )

    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("experiments/runs"),
        help="Directory containing per-run subdirectories.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/plots"),
        help="Directory to save plots into.",
    )

    parser.add_argument(
        "--basis",
        type=str,
        default="surface_code_X",
        help="Benchmark basis to filter on.",
    )

    parser.add_argument(
        "--decode-mode",
        type=str,
        default="batch",
        help="Decode mode to filter on.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Circuit worker count to filter on.",
    )

    parser.add_argument(
        "--n-shots",
        type=int,
        default=100000,
        help="Shot count to filter on.",
    )

    parser.add_argument(
        "--distances",
        type=int,
        nargs="*",
        default=[11],
        help="Optional distance filter.",
    )

    parser.add_argument(
        "--p-values",
        type=float,
        nargs="*",
        default=[0.001],
        help="Optional p-value filter.",
    )

    parser.add_argument(
        "--all-distances",
        action="store_true",
        help="Do not filter on distance values.",
    )

    parser.add_argument(
        "--all-p-values",
        action="store_true",
        help="Do not filter on p-values.",
    )

    args = parser.parse_args()

    if args.all_distances:
        args.distances = None
    else:
        args.distances = list(args.distances)

    if args.all_p_values:
        args.p_values = None
    else:
        args.p_values = list(args.p_values)

    run_dirs = expand_run_dirs([args.runs_root])
    runs = _collect_runs(run_dirs, args)

    if not runs:
        print(f"No matching thread-sweep runs found under {args.runs_root}")
        return 1

    print("threads,decode_time_seconds,shots_per_second,run_dir")
    for run in runs:
        print(
            f"{run.threads},"
            f"{run.decode_time_seconds:.6f},"
            f"{run.shots_per_second:.2f},"
            f"{run.run_dir.name}"
        )

    threads = [run.threads for run in runs]
    decode_times = [run.decode_time_seconds for run in runs]
    throughputs = [run.shots_per_second for run in runs]

    decode_time_png = args.output_dir / "thread_sweep_decode_time.png"
    throughput_png = args.output_dir / "thread_sweep_throughput.png"

    _plot_metric(
        threads,
        decode_times,
        xlabel="Decoder threads",
        ylabel="Decode time (s)",
        title="Thread sweep: decode time",
        output=decode_time_png,
    )

    _plot_metric(
        threads,
        throughputs,
        xlabel="Decoder threads",
        ylabel="Shots per second",
        title="Thread sweep: throughput",
        output=throughput_png,
    )

    print()
    print(f"Saved: {decode_time_png}")
    print(f"Saved: {throughput_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
