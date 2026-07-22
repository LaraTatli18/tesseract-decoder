from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class ParameterSweepPoint:
    parameter_value: float
    mean_decode_time_seconds: float
    mean_shots_per_second: float
    run_dirs: list[Path]
    manifests: list[dict[str, Any]]


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    with manifest_path.open() as f:
        return json.load(f)


def _load_result_rows(results_path: Path) -> list[dict[str, str]]:
    with results_path.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _as_float_list(values: list[Any]) -> list[float]:
    return [float(v) for v in values]


def _matches_filters(manifest: dict[str, Any], args: argparse.Namespace) -> bool:
    if args.basis is not None and manifest.get("basis") != args.basis:
        return False
    if args.decode_mode is not None and manifest.get("decode_mode") != args.decode_mode:
        return False
    if args.workers is not None and int(manifest.get("workers", -1)) != args.workers:
        return False
    if args.threads is not None and int(manifest.get("threads", -1)) != args.threads:
        return False
    if args.n_shots is not None and int(manifest.get("n_shots", -1)) != args.n_shots:
        return False
    if args.distances is not None and _as_float_list(manifest.get("distances", [])) != [float(d) for d in args.distances]:
        return False
    if args.p_values is not None and _as_float_list(manifest.get("p_values", [])) != [float(p) for p in args.p_values]:
        return False
    return True


def _check_manifests_consistent(
    run_records: list[tuple[Path, dict[str, Any]]],
    sweep_parameter: str,
) -> None:
    if not run_records:
        return

    reference_dir, reference = run_records[0]

    allowed = {
        "timestamp",
        "output_csv",
        "command",
        sweep_parameter,
    }

    for run_dir, manifest in run_records[1:]:
        for key, reference_value in reference.items():
            if key in allowed:
                continue
            current_value = manifest.get(key)
            if current_value != reference_value:
                raise ValueError(
                    f"Run directory {run_dir} differs in '{key}'.\n"
                    f"Reference run: {reference_dir}\n"
                    f"Reference value: {reference_value}\n"
                    f"Current value:   {current_value}"
                )


def _group_runs_by_parameter(
    run_dirs: list[Path],
    args: argparse.Namespace,
) -> tuple[dict[float, list[tuple[Path, dict[str, Any], list[dict[str, str]]]]], list[tuple[Path, dict[str, Any]]]]:
    grouped: dict[float, list[tuple[Path, dict[str, Any], list[dict[str, str]]]]] = {}
    run_records: list[tuple[Path, dict[str, Any]]] = []

    for run_dir in run_dirs:
        manifest_path = run_dir / "manifest.json"
        results_path = run_dir / "results.csv"

        if not manifest_path.exists() or not results_path.exists():
            print(f"Skipping {run_dir}: missing manifest.json or results.csv")
            continue

        try:
            manifest = _load_manifest(manifest_path)
        except json.JSONDecodeError:
            print(f"Skipping {run_dir}: invalid manifest.json")
            continue

        if not _matches_filters(manifest, args):
            continue

        if args.parameter not in manifest:
            print(f"Skipping {run_dir}: manifest missing '{args.parameter}'")
            continue

        try:
            param_value = float(manifest[args.parameter])
        except (TypeError, ValueError):
            print(f"Skipping {run_dir}: could not parse '{args.parameter}' as float")
            continue

        rows = _load_result_rows(results_path)
        if not rows:
            print(f"Skipping {run_dir}: results.csv is empty")
            continue

        grouped.setdefault(param_value, []).append((run_dir, manifest, rows))
        run_records.append((run_dir, manifest))

    return grouped, run_records


def _summarise_group(
    parameter_value: float,
    runs: list[tuple[Path, dict[str, Any], list[dict[str, str]]]],
) -> ParameterSweepPoint:
    ler_values: list[float] = []
    decode_time_values: list[float] = []
    throughput_values: list[float] = []
    run_dirs: list[Path] = []
    manifests: list[dict[str, Any]] = []

    for run_dir, manifest, rows in runs:
        run_dirs.append(run_dir)
        manifests.append(manifest)
        for row in rows:
            ler_values.append(float(row["logical_error_rate_per_round"]))
            decode_time_values.append(float(row["decode_time_seconds"]))
            throughput_values.append(float(row["shots_per_second"]))

    return ParameterSweepPoint(
        parameter_value=parameter_value,
        mean_decode_time_seconds=fmean(decode_time_values),
        mean_shots_per_second=fmean(throughput_values),
        run_dirs=run_dirs,
        manifests=manifests,
    )


def _save_plot(
    x: list[float],
    y: list[float],
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    output: Path,
    yscale: str = "linear",
) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(x, y, marker="o")
    for xi, yi in zip(x, y):
        plt.annotate(f"{xi:g}", (xi, yi), textcoords="offset points", xytext=(0, 8), ha="center")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.yscale(yscale)
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=300)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot a parameter sweep from run directories.")
    parser.add_argument(
        "--run-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="Run directories to compare.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/plots"),
        help="Directory to save the comparison plots into.",
    )
    parser.add_argument(
        "--parameter",
        type=str,
        default="det_beam",
        help="Manifest key to use as the sweep parameter.",
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
        "--threads",
        type=int,
        default=64,
        help="Decoder thread count to filter on.",
    )
    parser.add_argument(
        "--n-shots",
        type=int,
        default=None,
        help="Shot count to filter on.",
    )
    parser.add_argument(
        "--distances",
        type=int,
        nargs="*",
        default=None,
        help="Optional distance filter. If omitted, do not filter by distances.",
    )
    parser.add_argument(
        "--p-values",
        type=float,
        nargs="*",
        default=None,
        help="Optional p-value filter. If omitted, do not filter by p-values.",
    )
    args = parser.parse_args()

    # Expand experiment directories into individual run directories.
    expanded_run_dirs: list[Path] = []

    for path in args.run_dirs:
        manifest = path / "manifest.json"
        results = path / "results.csv"

        # If this is already a run directory, keep it.
        if manifest.exists() and results.exists():
            expanded_run_dirs.append(path)
            continue

        # Otherwise, treat it as an experiment directory and collect all runs.
        for child in sorted(path.iterdir()):
            if not child.is_dir():
                continue
            if (child / "manifest.json").exists() and (
                child / "results.csv").exists():
                expanded_run_dirs.append(child)

    args.run_dirs = expanded_run_dirs

    grouped, run_records = _group_runs_by_parameter(args.run_dirs, args)
    if not grouped:
        print("No matching runs found.")
        return 1

    _check_manifests_consistent(run_records, args.parameter)

    points = [_summarise_group(param, runs) for param, runs in sorted(grouped.items(), key=lambda kv: kv[0])]

    print("parameter,mean_decode_time_seconds,mean_shots_per_second,run_count")
    for point in points:
        print(
            f"{point.parameter_value:g},"
            f"{point.mean_decode_time_seconds:.6f},"
            f"{point.mean_shots_per_second:.2f},"
            f"{len(point.run_dirs)}"
        )

    x = [p.parameter_value for p in points]
    decode_time = [p.mean_decode_time_seconds for p in points]
    throughput = [p.mean_shots_per_second for p in points]

    shots = (
        f"{args.n_shots // 1_000_000}M"
        if args.n_shots and args.n_shots >= 1_000_000
        else f"{args.n_shots // 1_000}k"
        if args.n_shots
        else f"{int(points[0].manifests[0]['n_shots']) // 1000}k"
    )

    basis = points[0].manifests[0]["basis"]

    decode_time_png = (
        args.output_dir
        / f"{args.parameter}_{shots}_{basis}_decode_time.png"
    )

    throughput_png = (
        args.output_dir
        / f"{args.parameter}_{shots}_{basis}_throughput.png"
    )

    _save_plot(
        x,
        decode_time,
        xlabel=args.parameter,
        ylabel="Decode time (s)",
        title=f"{args.parameter} sweep: decode time",
        output=decode_time_png,
    )
    _save_plot(
        x,
        throughput,
        xlabel=args.parameter,
        ylabel="Shots per second",
        title=f"{args.parameter} sweep: throughput",
        output=throughput_png,
    )

    print()
    print(f"Saved: {decode_time_png}")
    print(f"Saved: {throughput_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
