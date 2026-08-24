from __future__ import annotations

"""Plot mean decoder runtime as a function of thread count.

This script compares three-repeat benchmark runs across decoder thread settings
and produces a plot of mean decode time versus thread count.
"""

import argparse
import re
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
    repeat: int
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

        repeat_match = re.search(
            r"thread_sweep_d11_p002_rep(\d+)",
            str(run.run_dir),
        )
        if repeat_match is None:
            print(
                f"Skipping {run.run_dir}: "
                "could not determine repeat number"
            )
            continue

        repeat = int(repeat_match.group(1))

        runs.append(
            ThreadSweepRun(
                threads=threads,
                decode_time_seconds=decode_time_seconds,
                shots_per_second=shots_per_second,
                repeat=repeat,
                run_dir=run.run_dir,
                manifest=run.manifest,
            )
        )

    runs.sort(key=lambda run: (run.threads, run.repeat))
    return runs


def _group_thread_metrics(
    runs: list[ThreadSweepRun],
) -> tuple[list[int], list[float], list[float]]:
    grouped_times: dict[int, list[float]] = {}
    grouped_throughput: dict[int, list[float]] = {}

    for run in runs:
        grouped_times.setdefault(run.threads, []).append(
            run.decode_time_seconds
        )
        grouped_throughput.setdefault(run.threads, []).append(
            run.shots_per_second
        )

    threads = sorted(grouped_times)

    mean_decode_times = [
        sum(grouped_times[t]) / len(grouped_times[t])
        for t in threads
    ]

    mean_throughput = [
        sum(grouped_throughput[t]) / len(grouped_throughput[t])
        for t in threads
    ]

    return threads, mean_decode_times, mean_throughput


def _plot_mean_decode_time(
    threads: list[int],
    means: list[float],
    *,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(threads, means, marker="o")

    for thread_count, mean_time in zip(threads, means):
        ax.annotate(
            str(thread_count),
            (thread_count, mean_time),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
        )

    ax.set_xlabel("Decoder threads")
    ax.set_ylabel("Mean decode time (s)")
    ax.set_title("Thread sweep: mean decode time")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure(fig, output)
    plt.close(fig)

def _plot_mean_throughput(
    threads: list[int],
    means: list[float],
    *,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(threads, means, marker="o")

    for thread_count, mean_throughput in zip(threads, means):
        ax.annotate(
            str(thread_count),
            (thread_count, mean_throughput),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
        )

    ax.set_xlabel("Decoder threads")
    ax.set_ylabel("Mean throughput (shots/s)")
    ax.set_title("Thread sweep: mean throughput")
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

    # The three repeats are run-group directories containing the
    # timestamped benchmark directories.
    repeat_groups = sorted(
        args.runs_root.glob("thread_sweep_d11_p002_rep*")
    )

    run_dirs: list[Path] = []

    for group_dir in repeat_groups:
        if group_dir.is_dir():
            run_dirs.extend(
                expand_run_dirs([group_dir])
            )

    runs = _collect_runs(run_dirs, args)

    if not runs:
        print(
            f"No matching thread-sweep runs found under "
            f"{args.runs_root}"
        )
        return 1

    print("threads,repeat,decode_time_seconds,run_dir")

    for run in runs:
        print(
            f"{run.threads},"
            f"{run.repeat},"
            f"{run.decode_time_seconds:.6f},"
            f"{run.run_dir.name}"
        )

    threads, mean_decode_times, mean_throughput = _group_thread_metrics(runs)

    decode_time_output = (
        args.output_dir
        / "thread_sweep_mean_decode_time.png"
    )

    throughput_output = (
        args.output_dir
        / "thread_sweep_mean_throughput.png"
    )

    _plot_mean_decode_time(
        threads,
        mean_decode_times,
        output=decode_time_output,
    )

    _plot_mean_throughput(
        threads,
        mean_throughput,
        output=throughput_output,
    )

    print()
    print(f"Saved: {decode_time_output}")
    print(f"Saved: {throughput_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
