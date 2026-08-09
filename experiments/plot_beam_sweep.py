from __future__ import annotations

"""Plot one-dimensional beam-parameter sweeps across physical error rates.

This script compares explicit benchmark run directories while varying one beam-related
manifest parameter (typically `det_beam` or `beam_climbing`). It produces one logical
error-rate-per-round versus physical-error-rate figure for each code distance, with
one curve per value of the swept parameter.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from plot_utils import (
    check_manifests_consistent,
    collect_plot_runs,
    create_plot_run,
    make_plot_filename,
    save_figure,
    half_shot_floor
)


@dataclass(frozen=True)
class BeamRun:
    run_dir: Path
    manifest: dict[str, Any]
    beam_value: float
    rows: list[dict[str, str]]


def _collect_runs(
    run_dirs: list[Path],
    parameter: str,
) -> list[BeamRun]:
    runs: list[BeamRun] = []

    for run in collect_plot_runs(run_dirs):
        if parameter not in run.manifest:
            print(
                f"Skipping {run.run_dir}: "
                f"manifest missing '{parameter}'"
            )
            continue

        try:
            beam_value = float(run.manifest[parameter])
        except (TypeError, ValueError):
            print(
                f"Skipping {run.run_dir}: "
                f"could not parse '{parameter}' as float"
            )
            continue

        runs.append(
            BeamRun(
                run_dir=run.run_dir,
                manifest=run.manifest,
                beam_value=beam_value,
                rows=run.rows,
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

    fig, ax = plt.subplots(figsize=(7, 4))

    floor = half_shot_floor(n_shots)

    for beam_value in sorted(grouped):
        rows = sorted(
            grouped[beam_value],
            key=lambda r: float(r["physical_error_rate"]),
        )

        x = [
            float(r["physical_error_rate"])
            for r in rows
        ]

        y = [
            max(
                float(r["logical_error_rate_per_round"]),
                floor,
            )
            for r in rows
        ]

        if parameter == "beam_climbing":
            label = (
                "Beam climbing ON"
                if beam_value
                else "Beam climbing OFF"
            )
        else:
            label = f"{parameter}={beam_value:g}"

        ax.plot(
            x,
            y,
            marker="o",
            label=label,
        )

    ax.set_xlabel("Physical error rate")
    ax.set_ylabel("Logical error rate per round")
    ax.set_yscale("log")
    ax.grid(
        True,
        which="both",
        linestyle="--",
        alpha=0.4,
    )
    ax.legend()
    ax.set_title(
        f"{basis} | d={distance} | {n_shots:,} shots"
    )

    fig.tight_layout()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out = output_dir / make_plot_filename(
        parameter,
        basis,
        n_shots,
        distance,
    )

    save_figure(fig, out)
    plt.close(fig)

    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot beam sweeps from explicit run directories."
    )

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

    runs = _collect_runs(
        args.run_dirs,
        args.parameter,
    )

    if not runs:
        print("No matching runs found.")
        return 1

    manifests = [
        run.manifest
        for run in runs
    ]

    check_manifests_consistent(
        manifests,
        varying_fields={args.parameter},
    )

    reference = runs[0].manifest
    basis = reference["basis"]
    n_shots = int(reference["n_shots"])

    distances = sorted(
        {
            int(row["distance"])
            for run in runs
            for row in run.rows
        }
    )

    print("distance,beam,rows")

    for distance in distances:
        for run in sorted(
            runs,
            key=lambda r: r.beam_value,
        ):
            row_count = sum(
                1
                for row in run.rows
                if int(row["distance"]) == distance
            )

            if row_count:
                print(
                    f"{distance},"
                    f"{run.beam_value:g},"
                    f"{row_count}"
                )

    output_dir = create_plot_run(
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
