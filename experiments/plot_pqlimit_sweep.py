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


def _geometric_edges(
    values: list[float],
) -> np.ndarray:
    vals = sorted(values)

    if len(vals) == 1:
        v = vals[0]
        return np.array(
            [
                v / np.sqrt(10.0),
                v * np.sqrt(10.0),
            ]
        )

    edges = [
        vals[0]
        / np.sqrt(vals[1] / vals[0])
    ]

    for i in range(1, len(vals)):
        edges.append(
            np.sqrt(
                vals[i - 1] * vals[i]
            )
        )

    edges.append(
        vals[-1]
        * np.sqrt(
            vals[-1] / vals[-2]
        )
    )

    return np.array(edges)


def _linear_edges(
    values: list[float],
) -> np.ndarray:
    vals = sorted(values)

    if len(vals) == 1:
        v = vals[0]
        return np.array(
            [
                v - 0.5,
                v + 0.5,
            ]
        )

    edges = [
        vals[0]
        - (vals[1] - vals[0]) / 2
    ]

    for i in range(1, len(vals)):
        edges.append(
            (
                vals[i - 1]
                + vals[i]
            )
            / 2
        )

    edges.append(
        vals[-1]
        + (
            vals[-1]
            - vals[-2]
        )
        / 2
    )

    return np.array(edges)


def _compute_global_heatmap_scale(
    aggregated: dict[
        tuple[int, float, float, float],
        dict[str, float],
    ],
) -> tuple[float, float]:
    values: list[float] = []

    for stats in aggregated.values():
        n_shots = int(
            stats["n_shots"]
        )

        floor = half_shot_floor(n_shots)

        value = max(
            stats[
                "logical_error_rate_per_round"
            ],
            floor,
        )

        values.append(
            np.log10(value)
        )

    if not values:
        raise ValueError(
            "No data available to compute heatmap scale"
        )

    vmin = min(values)
    vmax = max(values)

    if vmin == vmax:
        vmin -= 1e-12
        vmax += 1e-12

    return vmin, vmax


def _plot_line_cross_sections(
    aggregated: dict[
        tuple[int, float, float, float],
        dict[str, float],
    ],
    *,
    p_value: float,
    basis: str,
    output_dir: Path,
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
            9.5,
            2.5 * len(distances),
        ),
        sharex=True,
        sharey=True,
    )

    if len(distances) == 1:
        axes = [axes]

    colours = list(
        plt.get_cmap("tab10").colors
    )

    colour_for_beam = {
        beam: colours[
            i % len(colours)
        ]
        for i, beam
        in enumerate(beams)
    }

    n_shots = int(
        next(
            iter(
                aggregated.values()
            )
        )["n_shots"]
    )

    floor = half_shot_floor(n_shots)

    for ax, distance in zip(
        axes,
        distances,
    ):
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

                ys.append(
                    max(
                        aggregated[key][
                            "logical_error_rate_per_round"
                        ],
                        floor,
                    )
                )

            if xs:
                ax.plot(
                    xs,
                    ys,
                    marker="o",
                    markersize=4,
                    linewidth=2,
                    color=colour_for_beam[
                        beam
                    ],
                    label=(
                        f"Beam "
                        f"{int(beam)}"
                    ),
                )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.minorticks_on()

        ax.grid(
            True,
            which="major",
            linestyle="--",
            alpha=0.3,
        )

        ax.grid(
            True,
            which="minor",
            linestyle=":",
            alpha=0.15,
        )

        ax.set_title(
            f"d = {distance}",
            fontsize=12,
        )

    axes[-1].set_xlabel(
        "Priority queue limit"
    )

    axes[0].set_ylabel(
        "Logical error rate per round"
    )

    handles, labels = (
        axes[0]
        .get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            0.99,
        ),
        ncol=3,
        frameon=False,
        fontsize=9,
    )

    fig.suptitle(
        f"{basis} – "
        f"Priority queue sweep "
        f"(p = {p_value:g})",
        fontsize=14,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.95,
        ]
    )

    out = (
        output_dir
        / (
            f"{basis}_"
            f"{format_p_value(p_value)}_"
            f"line_cross_sections.png"
        )
    )

    save_figure(
        fig,
        out,
    )

    plt.close(fig)

    return out


def _plot_heatmaps(
    aggregated: dict[
        tuple[int, float, float, float],
        dict[str, float],
    ],
    *,
    p_value: float,
    basis: str,
    output_dir: Path,
    abs_vmin: float,
    abs_vmax: float,
    relative: bool = False,
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
            8.5,
            2.8 * len(distances),
        ),
        sharex=True,
    )

    if len(distances) == 1:
        axes = [axes]

    cmap = (
        plt.get_cmap("magma")
        .copy()
    )

    cmap.set_bad(
        "lightgrey"
    )

    x_edges = _geometric_edges(
        pqlimits
    )

    y_edges = _linear_edges(
        beams
    )

    last_mesh = None

    for ax, distance in zip(
        axes,
        distances,
    ):
        matrix = np.full(
            (
                len(beams),
                len(pqlimits),
            ),
            np.nan,
            dtype=float,
        )

        values: list[float] = []

        for beam in beams:
            for pq in pqlimits:
                key = (
                    distance,
                    p_value,
                    beam,
                    pq,
                )

                if key not in aggregated:
                    continue

                stats = aggregated[key]

                floor = half_shot_floor(stats["n_shots"])

                values.append(
                    max(
                        stats[
                            "logical_error_rate_per_round"
                        ],
                        floor,
                    )
                )

        best = min(values)

        for i, beam in enumerate(
            beams
        ):
            for j, pq in enumerate(
                pqlimits
            ):
                key = (
                    distance,
                    p_value,
                    beam,
                    pq,
                )

                if key not in aggregated:
                    continue

                stats = aggregated[key]

                floor = half_shot_floor(stats["n_shots"])

                value = max(
                    stats[
                        "logical_error_rate_per_round"
                    ],
                    floor,
                )

                if relative:
                    matrix[i, j] = (
                        np.log10(
                            value / best
                        )
                    )
                else:
                    matrix[i, j] = (
                        np.log10(
                            value
                        )
                    )

        masked = (
            np.ma.masked_invalid(
                matrix
            )
        )

        if relative:
            last_mesh = (
                ax.pcolormesh(
                    x_edges,
                    y_edges,
                    masked,
                    shading="auto",
                    cmap=cmap,
                    vmin=0,
                    vmax=np.nanmax(
                        masked
                    ),
                )
            )
        else:
            last_mesh = (
                ax.pcolormesh(
                    x_edges,
                    y_edges,
                    masked,
                    shading="auto",
                    cmap=cmap,
                    vmin=abs_vmin,
                    vmax=abs_vmax,
                )
            )

        ax.set_xscale("log")
        ax.set_ylabel("det_beam")
        ax.set_yticks(beams)

        ax.set_yticklabels(
            [
                f"{beam:g}"
                for beam in beams
            ]
        )

        ax.set_title(
            f"d={distance}, "
            f"p={p_value:g}",
            fontsize=11,
        )

        ax.grid(False)

    axes[-1].set_xlabel(
        "Priority queue limit"
    )

    fig.subplots_adjust(
        right=0.86
    )

    cbar = fig.colorbar(
        last_mesh,
        ax=axes,
        fraction=0.04,
        pad=0.02,
    )

    if relative:
        cbar.set_label(
            r"$\log_{10}"
            r"(\mathrm{LER}/"
            r"\mathrm{Best\ LER})$"
        )
    else:
        cbar.set_label(
            r"$\log_{10}"
            r"(\mathrm{Logical\ Error\ Rate"
            r"\ per\ Round})$"
        )

    fig.suptitle(
        f"{basis} | "
        f"p={p_value:g} | "
        f"heatmaps",
        y=0.995,
    )

    suffix = (
        "relative_heatmaps"
        if relative
        else "heatmaps"
    )

    out = (
        output_dir
        / (
            f"{basis}_"
            f"{format_p_value(p_value)}_"
            f"{suffix}.png"
        )
    )

    save_figure(
        fig,
        out,
    )

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
        default=64,
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

    aggregated = _aggregate_runs(
        runs
    )

    abs_vmin, abs_vmax = (
        _compute_global_heatmap_scale(
            aggregated
        )
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
        line_png = (
            _plot_line_cross_sections(
                aggregated,
                p_value=p_value,
                basis=args.basis,
                output_dir=output_dir,
            )
        )

        heatmap_png = (
            _plot_heatmaps(
                aggregated,
                p_value=p_value,
                basis=args.basis,
                output_dir=output_dir,
                abs_vmin=abs_vmin,
                abs_vmax=abs_vmax,
                relative=False,
            )
        )

        relative_png = (
            _plot_heatmaps(
                aggregated,
                p_value=p_value,
                basis=args.basis,
                output_dir=output_dir,
                abs_vmin=abs_vmin,
                abs_vmax=abs_vmax,
                relative=True,
            )
        )

        print(
            f"{p_value:g},"
            f"line,"
            f"{line_png}"
        )

        print(
            f"{p_value:g},"
            f"heatmap,"
            f"{heatmap_png}"
        )

        print(
            f"{p_value:g},"
            f"relative_heatmap,"
            f"{relative_png}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
