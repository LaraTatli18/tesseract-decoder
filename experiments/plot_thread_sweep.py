from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class ThreadSweepPoint:
    threads: int
    decode_time_seconds: float
    shots_per_second: float
    run_dir: Path
    manifest: dict[str, Any]


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    with manifest_path.open() as f:
        return json.load(f)


def _load_first_result_row(results_path: Path) -> dict[str, str] | None:
    with results_path.open(newline="") as f:
        reader = csv.DictReader(f)
        try:
            return next(reader)
        except StopIteration:
            return None


def _matches_filters(manifest: dict[str, Any], args: argparse.Namespace) -> bool:
    """Return True if a run manifest matches the requested sweep filters."""

    if args.decode_mode is not None and manifest.get("decode_mode") != args.decode_mode:
        return False
    if args.basis is not None and manifest.get("basis") != args.basis:
        return False
    if args.workers is not None and int(manifest.get("workers", -1)) != args.workers:
        return False
    if args.n_shots is not None and int(manifest.get("n_shots", -1)) != args.n_shots:
        return False
    if args.distances is not None and list(manifest.get("distances", [])) != args.distances:
        return False
    if args.p_values is not None and list(manifest.get("p_values", [])) != args.p_values:
        return False
    return True


def _collect_points(runs_root: Path, args: argparse.Namespace) -> list[ThreadSweepPoint]:
    points: list[ThreadSweepPoint] = []

    for manifest_path in sorted(runs_root.glob("*/manifest.json")):
        run_dir = manifest_path.parent
        results_path = run_dir / "results.csv"
        if not results_path.exists():
            continue

        try:
            manifest = _load_manifest(manifest_path)
        except json.JSONDecodeError:
            continue

        if not _matches_filters(manifest, args):
            continue

        row = _load_first_result_row(results_path)
        if row is None:
            continue

        try:
            threads = int(manifest["threads"])
            decode_time_seconds = float(row["decode_time_seconds"])
            shots_per_second = float(row["shots_per_second"])
        except (KeyError, TypeError, ValueError):
            continue

        points.append(
            ThreadSweepPoint(
                threads=threads,
                decode_time_seconds=decode_time_seconds,
                shots_per_second=shots_per_second,
                run_dir=run_dir,
                manifest=manifest,
            )
        )

    points.sort(key=lambda p: p.threads)
    return points


def _save_plot(
    x: list[int],
    y: list[float],
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    output: Path,
    annotate_points: bool = True,
) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(x, y, marker="o")
    if annotate_points:
        for xi, yi in zip(x, y):
            plt.annotate(str(xi), (xi, yi), textcoords="offset points", xytext=(0, 8), ha="center")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot results from a decoder thread sweep.")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("experiments/runs"),
        help="Directory containing per-run subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/runs"),
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
        help="Optional distance filter. Use the full list from the sweep if desired.",
    )
    parser.add_argument(
        "--p-values",
        type=float,
        nargs="*",
        default=[0.001],
        help="Optional p-value filter. Use the exact values from the sweep if desired.",
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

    points = _collect_points(args.runs_root, args)
    if not points:
        print(f"No matching thread-sweep runs found under {args.runs_root}")
        return 1

    print("threads,decode_time_seconds,shots_per_second,run_dir")
    for p in points:
        print(f"{p.threads},{p.decode_time_seconds:.6f},{p.shots_per_second:.2f},{p.run_dir.name}")

    threads = [p.threads for p in points]
    decode_times = [p.decode_time_seconds for p in points]
    throughputs = [p.shots_per_second for p in points]

    decode_time_png = args.output_dir / "thread_sweep_decode_time.png"
    throughput_png = args.output_dir / "thread_sweep_throughput.png"

    _save_plot(
        threads,
        decode_times,
        xlabel="Decoder threads",
        ylabel="Decode time (s)",
        title="Thread sweep: decode time",
        output=decode_time_png,
    )
    _save_plot(
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
