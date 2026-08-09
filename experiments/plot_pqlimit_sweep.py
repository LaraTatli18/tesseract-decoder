from __future__ import annotations

"""Plot priority-queue-limit sweeps over physical error rate.

This script compares explicit benchmark run directories while varying pqlimit and
beam-related manifest settings. It produces line-cross-section plots and heatmaps
for logical error rate per round as a function of priority queue limit, detector
beam, and distance.
"""

from collections import defaultdict
import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LINE_METRICS = [
    ("logical_error_rate_per_round", "Logical error rate per round", "logical_error_rate", True),
    ("decode_time_seconds", "Decode time (s)", "decode_time", False),
    ("shots_per_second", "Shots per second", "throughput", False),
]

from plot_utils import (
    collect_plot_runs,
    expand_run_dirs,
    matches_filters,
    save_figure,
    setup_matplotlib,
    half_shot_floor,
    format_p_value
)


@dataclass(frozen=True)
class PQLimitSweepRun:
    run_dir: Path
    distance: int
    physical_error_rate: float
    det_beam: float
    pqlimit: float
    logical_error_rate_per_round: float
    decode_time_seconds: float
    shots_per_second: float
    n_shots: int


def _collect_runs(
    run_dirs: list[Path],
    args: argparse.Namespace,
) -> list[PQLimitSweepRun]:
    runs: list[PQLimitSweepRun] = []

    for run in collect_plot_runs(run_dirs):
        if not matches_filters(run.manifest, args):
            continue

        try:
            det_beam = float(run.manifest["det_beam"])
            pqlimit = float(run.manifest["pqlimit"])
        except (KeyError, TypeError, ValueError):
            print(
                f"Skipping {run.run_dir}: "
                "missing or invalid det_beam/pqlimit in manifest"
            )
            continue

        for row in run.rows:
            try:
                runs.append(
                    PQLimitSweepRun(
                        run_dir=run.run_dir,
                        distance=int(row["distance"]),
                        physical_error_rate=float(
                            row["physical_error_rate"]
                        ),
                        det_beam=det_beam,
                        pqlimit=pqlimit,
                        logical_error_rate_per_round=float(
                            row["logical_error_rate_per_round"]
                        ),
                        decode_time_seconds=float(
                            row["decode_time_seconds"]
                        ),
                        shots_per_second=float(
                            row["shots_per_second"]
                        ),
                        n_shots=int(row["n_shots"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                print(
                    f"Skipping malformed row in "
                    f"{run.run_dir / 'results.csv'}"
                )
                continue

    return runs


def _keep_latest_runs(
    runs: list[PQLimitSweepRun],
) -> list[PQLimitSweepRun]:
    """Keep only the latest complete run for each benchmark configuration. This is useful if we have duplicates."""
    latest: dict[
        tuple[int, float, float, float],
        PQLimitSweepRun,
    ] = {}

    for run in runs:
        key = (
            run.distance,
            run.physical_error_rate,
            run.det_beam,
            run.pqlimit,
        )

        previous = latest.get(key)

        if previous is None or run.run_dir.name > previous.run_dir.name:
            latest[key] = run

    removed = len(runs) - len(latest)

    if removed:
        print(
            f"Collapsed {removed} duplicate result rows "
            "by keeping the latest run for each configuration."
        )

    return list(latest.values())


def _aggregate_runs(
    runs: list[PQLimitSweepRun],
) -> dict[tuple[int, float, float, float], dict[str, float]]:
    grouped: dict[
        tuple[int, float, float, float],
        list[PQLimitSweepRun],
    ] = defaultdict(list)

    for run in runs:
        key = (
            run.distance,
            run.physical_error_rate,
            run.det_beam,
            run.pqlimit,
        )
        grouped[key].append(run)

    aggregated: dict[
        tuple[int, float, float, float],
        dict[str, float],
    ] = {}

    for key, items in grouped.items():
        aggregated[key] = {
            "logical_error_rate_per_round": float(
                np.mean(
                    [
                        run.logical_error_rate_per_round
                        for run in items
                    ]
                )
            ),
            "decode_time_seconds": float(
                np.mean(
                    [
                        run.decode_time_seconds
                        for run in items
                    ]
                )
            ),
            "shots_per_second": float(
                np.mean(
                    [
                        run.shots_per_second
                        for run in items
                    ]
                )
            ),
            "n_shots": float(items[0].n_shots),
        }

    return aggregated

def _plot_line_cross_sections(
    aggregated: dict[
        tuple[int, float, float, float],
        dict[str, float],
    ],
    *,
    p_value: float,
    basis: str,
    output_dir: Path,
    metric_key: str,
    metric_label: str,
    metric_slug: str,
    floor_to_half_shot: bool = False,
    yscale: str = "log",
) -> Path:
    distances = sorted(
        {
            distance
            for (
                distance,
                p,
                _,
                _,
            ) in aggregated
            if p == p_value
        }
    )

    beams = sorted(
        {
            beam
            for (
                _,
                p,
                beam,
                _,
            ) in aggregated
            if p == p_value
        }
    )

    pqlimits = sorted(
        {
            pq
            for (
                _,
                p,
                _,
                pq,
            ) in aggregated
            if p == p_value
        }
    )

    if not distances:
        raise ValueError(
            f"No data found for p={p_value:g}"
        )

    fig, axes = plt.subplots(
        nrows=len(distances),
        ncols=1,
        figsize=(
            10,
            4.0 * len(distances),
        ),
        sharex=True,
        sharey=False,
    )

    if len(distances) == 1:
        axes = [axes]

    colours = list(plt.get_cmap("tab10").colors)
    colour_for_beam = {
        beam: colours[i % len(colours)]
        for i, beam in enumerate(beams)
    }

    n_shots = int(
        next(iter(aggregated.values()))["n_shots"]
    )

    floor = half_shot_floor(n_shots) if floor_to_half_shot else None

    for ax, distance in zip(axes, distances):
        for beam in beams:
            xs: list[float] = []
            ys: list[float] = []

            for pq in pqlimits:
                key = (
                    distance,
                    p_value,
                    beam,
                    pq,
                )

                if key not in aggregated:
                    continue

                xs.append(pq)

                value = aggregated[key][metric_key]
                if floor is not None:
                    value = max(value, floor)
                ys.append(value)

            if xs:
                ax.plot(
                    xs,
                    ys,
                    marker="o",
                    markersize=4,
                    linewidth=2,
                    color=colour_for_beam[beam],
                    label=f"Beam {int(beam)}",
                )

        ax.set_xscale("log")
        ax.set_yscale(yscale)
        ax.minorticks_on()
        ax.grid(True, which="major", linestyle="--", alpha=0.3)
        ax.grid(True, which="minor", linestyle=":", alpha=0.15)
        ax.set_title(f"d = {distance}", fontsize=12)

    axes[-1].set_xlabel("Priority queue limit")
    axes[0].set_ylabel(metric_label)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=3,
        frameon=False,
        fontsize=9,
    )

    fig.suptitle(
        f"{basis} – Priority queue sweep ({metric_label}, p = {p_value:g})",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out = (
        output_dir
        / (
            f"{basis}_"
            f"{format_p_value(p_value)}_"
            f"{metric_slug}_line_cross_sections.png"
        )
    )

    save_figure(fig, out)
    plt.close(fig)
    return out

def main() -> int:
    setup_matplotlib()

    parser = argparse.ArgumentParser(
        description=(
            "Plot the pqlimit sweep."
        )
    )

    parser.add_argument(
        "--run-dirs",
        type=Path,
        nargs="+",
        required=True,
        help=(
            "Run directories or "
            "experiment directories "
            "to compare."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "experiments/plots/"
            "pqlimit_sweep"
        ),
        help=(
            "Directory to save "
            "the plots into."
        ),
    )

    parser.add_argument(
        "--basis",
        type=str,
        default="surface_code_X",
        help="Basis to filter on.",
    )

    parser.add_argument(
        "--decode-mode",
        type=str,
        default="batch",
        help=(
            "Decode mode "
            "to filter on."
        ),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Circuit worker count "
            "to filter on."
        ),
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help=(
            "Decoder thread count "
            "to filter on."
        ),
    )

    parser.add_argument(
        "--n-shots",
        type=int,
        default=None,
        help=(
            "Shot count "
            "to filter on."
        ),
    )

    parser.add_argument(
        "--distances",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Optional distance filter. "
            "If omitted, do not filter "
            "by distances."
        ),
    )

    parser.add_argument(
        "--p-values",
        type=float,
        nargs="*",
        default=None,
        help=(
            "Optional p-value filter. "
            "If omitted, do not filter "
            "by p-values."
        ),
    )

    parser.add_argument(
        "--beam-climbing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Filter beam climbing "
            "on/off."
        ),
    )

    parser.add_argument(
        "--merge-errors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Filter merge_errors "
            "on/off."
        ),
    )

    args = parser.parse_args()

    run_dirs = expand_run_dirs(
        args.run_dirs
    )

    if not run_dirs:
        print(
            "No run directories found."
        )
        return 1

    runs = _collect_runs(
        run_dirs,
        args,
    )

    if not runs:
        print(
            "No matching rows found."
        )
        return 1

    runs = _keep_latest_runs(runs)

    aggregated = _aggregate_runs(
        runs
    )

    output_dir = args.output_dir

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    p_values = sorted(
        {
            run.physical_error_rate
            for run in runs
        }
    )

    print(
        "p_value,"
        "figure_type,"
        "output_file"
    )

    for p_value in p_values:
        for metric_key, metric_label, metric_slug, floor_to_half_shot in LINE_METRICS:
            line_png = _plot_line_cross_sections(
                aggregated,
                p_value=p_value,
                basis=args.basis,
                output_dir=output_dir,
                metric_key=metric_key,
                metric_label=metric_label,
                metric_slug=metric_slug,
                floor_to_half_shot=floor_to_half_shot,
                yscale="log",
            )
            print(
                f"{p_value:g},"
                f"{metric_slug},"
                f"{line_png}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
