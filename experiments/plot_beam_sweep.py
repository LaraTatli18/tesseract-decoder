from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class BeamRun:
    run_dir: Path
    manifest: dict[str, Any]
    beam_value: float
    rows: list[dict[str, str]]


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    with manifest_path.open() as f:
        return json.load(f)


def _load_result_rows(results_path: Path) -> list[dict[str, str]]:
    with results_path.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)

def _format_shot_count(n_shots: int) -> str:
    if n_shots >= 1_000_000:
        return f"{n_shots // 1_000_000}M"
    if n_shots >= 1_000:
        return f"{n_shots // 1_000}k"
    return str(n_shots)


def _create_plot_run(
    parameter: str,
    basis: str,
    n_shots: int,
    run_dirs: list[Path],
    script_name: str,
) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    shots = _format_shot_count(n_shots)

    output_dir = (
        Path("experiments/plots")
        / f"{timestamp}_{parameter}_{shots}_{basis}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    manifest = {
        "timestamp": timestamp,
        "plot_script": script_name,
        "parameter": parameter,
        "basis": basis,
        "n_shots": n_shots,
        "run_dirs": [str(r) for r in run_dirs],
    }

    with (output_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=4)

    return output_dir


def _make_plot_filename(
    parameter: str,
    basis: str,
    n_shots: int,
    distance: int,
) -> str:
    shots = _format_shot_count(n_shots)
    return f"{parameter}_{shots}_{basis}_d{distance}.png"


def _check_manifests_consistent(
    runs: list[BeamRun],
    sweep_parameter: str,
) -> None:
    if not runs:
        return

    reference_run = runs[0]
    reference = reference_run.manifest

    allowed = {
        "timestamp",
        "output_csv",
        "command",
        sweep_parameter,
    }

    reference_keys = set(reference.keys()) - allowed

    for run in runs[1:]:
        manifest = run.manifest
        for key in reference_keys:
            if manifest.get(key) != reference.get(key):
                raise ValueError(
                    f"Run directory {run.run_dir} differs in '{key}'.\n"
                    f"Reference run: {reference_run.run_dir}\n"
                    f"Reference value: {reference.get(key)}\n"
                    f"Current value:   {manifest.get(key)}"
                )


def _collect_runs(run_dirs: list[Path], parameter: str) -> list[BeamRun]:
    runs: list[BeamRun] = []

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

        if parameter not in manifest:
            print(f"Skipping {run_dir}: manifest missing '{parameter}'")
            continue

        try:
            beam_value = float(manifest[parameter])
        except (TypeError, ValueError):
            print(f"Skipping {run_dir}: could not parse '{parameter}' as float")
            continue

        rows = _load_result_rows(results_path)
        if not rows:
            print(f"Skipping {run_dir}: results.csv is empty")
            continue

        runs.append(
            BeamRun(
                run_dir=run_dir,
                manifest=manifest,
                beam_value=beam_value,
                rows=rows,
            )
        )

    return runs


def _plot_distance(
    distance: int,
    runs: list[BeamRun],
    parameter: str,
    output_dir: Path,
    n_shots: int,
    basis: str,
) -> Path:
    grouped: dict[float, list[dict[str, str]]] = {}

    for run in runs:
        for row in run.rows:
            if int(row["distance"]) != distance:
                continue
            grouped.setdefault(run.beam_value, []).append(row)

    if not grouped:
        raise ValueError(f"No rows found for distance d={distance}")

    plt.figure(figsize=(7, 4))

    floor = 0.5 / n_shots

    for beam_value in sorted(grouped):
        rows = sorted(grouped[beam_value], key=lambda r: float(r["physical_error_rate"]))
        x = [float(r["physical_error_rate"]) for r in rows]
        y = [
            max(float(r["logical_error_rate_per_round"]), floor)
            for r in rows
        ]

        if parameter == "beam_climbing":
            label = "Beam climbing ON" if beam_value else "Beam climbing OFF"
        else:
            label = f"{parameter}={beam_value:g}"

        plt.plot(x, y, marker="o", label=label)

    plt.xlabel("Physical error rate")
    plt.ylabel("Logical error rate per round")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.title(f"{basis} | d={distance} | {n_shots:,} shots")
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / _make_plot_filename(
        parameter,
        basis,
        n_shots,
        distance,
    )
    plt.savefig(out, dpi=300)
    plt.close()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot beam sweeps from explicit run directories.")
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
        default=Path("experiments/plots/beam_sweep"),
        help="Directory to save the comparison plots into.",
    )
    parser.add_argument(
        "--parameter",
        type=str,
        default="det_beam",
        help="Manifest key to use as the sweep parameter.",
    )
    args = parser.parse_args()

    runs = _collect_runs(args.run_dirs, args.parameter)
    if not runs:
        print("No matching runs found.")
        return 1

    _check_manifests_consistent(runs, args.parameter)

    reference = runs[0].manifest
    basis = reference["basis"]
    n_shots = int(reference["n_shots"])

    distances = sorted({int(row["distance"]) for run in runs for row in run.rows})

    print("distance,beam,rows")
    for distance in distances:
        for run in sorted(runs, key=lambda r: r.beam_value):
            row_count = sum(1 for row in run.rows if int(row["distance"]) == distance)
            if row_count:
                print(f"{distance},{run.beam_value:g},{row_count}")

    output_dir = _create_plot_run(
        parameter=args.parameter,
        basis=basis,
        n_shots=n_shots,
        run_dirs=args.run_dirs,
        script_name=Path(__file__).name,
    )

    saved_paths: list[Path] = []

    for distance in distances:
        saved_paths.append(
            _plot_distance(
                distance=distance,
                runs=runs,
                parameter=args.parameter,
                output_dir=output_dir,
                n_shots=n_shots,
                basis=basis,
            )
        )

    print()
    for path in saved_paths:
        print(f"Saved: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
