from __future__ import annotations

"""Plot the two-dimensional detector-beam / beam-climbing parameter study.

This script loads benchmark runs spanning detector-beam values, priority-queue
limits, and beam-climbing states. It compares logical error rate, decode time,
and throughput, and optionally produces detector-beam versus priority-queue
heatmaps and beam-climbing benefit/penalty maps.

Repeated executions of the same nominal configuration are aggregated before
plotting.
"""

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, TwoSlopeNorm

from plot_utils import (
    check_manifests_consistent,
    collect_plot_runs,
    expand_run_dirs,
    format_shot_count,
    parse_bool,
    save_figure,
    half_shot_floor,
)


@dataclass(frozen=True)
class Beam2DRun:
    run_dir: Path
    manifest: dict[str, Any]
    det_beam: float
    pqlimit: int
    beam_climbing: bool
    rows: list[dict[str, str]]


METRICS = {
    "logical_error_rate_per_round": {
        "ylabel": "Logical error rate per round",
        "yscale": "log",
        "slug": "logical_error_rate",
        "floor": True,
    },
    "decode_time_seconds": {
        "ylabel": "Decode time (s)",
        "yscale": "linear",
        "slug": "decode_time",
        "floor": False,
    },
    "shots_per_second": {
        "ylabel": "Shots per second",
        "yscale": "linear",
        "slug": "throughput",
        "floor": False,
    },
}


def _collect_runs(
    run_dirs: list[Path],
    args: argparse.Namespace,
) -> list[Beam2DRun]:
    runs: list[Beam2DRun] = []

    for run in collect_plot_runs(run_dirs):
        manifest = run.manifest

        # Scalar filters.
        if (
            args.basis is not None
            and manifest.get("basis") != args.basis
        ):
            continue

        if (
            args.decode_mode is not None
            and manifest.get("decode_mode") != args.decode_mode
        ):
            continue

        if (
            args.workers is not None
            and manifest.get("workers") != args.workers
        ):
            continue

        if (
            args.threads is not None
            and manifest.get("threads") != args.threads
        ):
            continue

        if (
            args.n_shots is not None
            and manifest.get("n_shots") != args.n_shots
        ):
            continue

        # Distance filter.
        #
        # Each publication manifest usually contains one distance, e.g. [9],
        # while the plotting command may request several distances, e.g. [9, 11].
        # Accept the run if its manifest contains ANY requested distance.
        manifest_distances = set(
            manifest.get("distances", [])
        )

        if args.distances is not None:
            requested_distances = set(
                args.distances
            )

            if not manifest_distances.intersection(
                requested_distances
            ):
                continue

        # Physical-error-rate filter.
        #
        # Likewise, accept a run if its manifest contains any requested p value.
        manifest_p_values = set(
            manifest.get("p_values", [])
        )

        if args.p_values is not None:
            requested_p_values = set(
                args.p_values
            )

            if not manifest_p_values.intersection(
                requested_p_values
            ):
                continue

        try:
            det_beam = float(manifest["det_beam"])

        except (KeyError, TypeError, ValueError):
            print(
                f"Skipping {run.run_dir}: "
                "could not parse 'det_beam' as float"
            )
            continue

        try:
            pqlimit = int(manifest["pqlimit"])
        except (KeyError, TypeError, ValueError):
            print(
                f"Skipping {run.run_dir}: "
                "could not parse 'pqlimit' as int"
            )
            continue

        try:
            beam_climbing = parse_bool(
                manifest["beam_climbing"]
            )
        except (KeyError, ValueError):
            print(
                f"Skipping {run.run_dir}: "
                "could not parse 'beam_climbing' as bool"
            )
            continue

        runs.append(
            Beam2DRun(
                run_dir=run.run_dir,
                manifest=manifest,
                det_beam=det_beam,
                pqlimit=pqlimit,
                beam_climbing=beam_climbing,
                rows=run.rows,
            )
        )

    return runs


def _flatten_rows(
    runs: list[Beam2DRun],
) -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []

    for run in runs:
        for row in run.rows:
            try:
                logical_failures = int(
                    row["logical_failures"]
                )
            except (KeyError, TypeError, ValueError):
                logical_failures = None

            rows_out.append(
                {
                    "run_dir": str(run.run_dir),
                    "det_beam": run.det_beam,
                    "pqlimit": run.pqlimit,
                    "beam_climbing": run.beam_climbing,
                    "distance": int(row["distance"]),
                    "physical_error_rate": float(
                        row["physical_error_rate"]
                    ),
                    "logical_failures": logical_failures,
                    "n_shots": int(
                        row.get(
                            "n_shots",
                            run.manifest["n_shots"],
                        )
                    ),
                    "logical_error_rate_per_round": float(
                        row["logical_error_rate_per_round"]
                    ),
                    "decode_time_seconds": float(
                        row["decode_time_seconds"]
                    ),
                    "shots_per_second": float(
                        row["shots_per_second"]
                    ),
                }
            )

    return rows_out


def _aggregate_configuration_rows(
    raw_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[int, float, float, int, bool],
        list[dict[str, Any]],
    ] = {}

    for row in raw_rows:
        key = (
            row["distance"],
            row["physical_error_rate"],
            row["det_beam"],
            row["pqlimit"],
            row["beam_climbing"],
        )

        grouped.setdefault(
            key,
            [],
        ).append(row)

    aggregated: list[dict[str, Any]] = []

    for key, group in grouped.items():
        (
            distance,
            p_value,
            det_beam,
            pqlimit,
            beam_climbing,
        ) = key

        total_shots = sum(
            row["n_shots"]
            for row in group
        )

        known_failures = [
            row["logical_failures"]
            for row in group
            if row["logical_failures"] is not None
        ]

        if known_failures:
            total_failures = sum(
                known_failures
            )
        else:
            total_failures = None

        # The plotted quantity is logical error rate PER ROUND.
        # Without separately storing the number of rounds, use a
        # shot-weighted average of the already-computed per-round rates.
        pooled_logical_error_rate = (
            sum(
                row["logical_error_rate_per_round"]
                * row["n_shots"]
                for row in group
            )
            / total_shots
            if total_shots > 0
            else float("nan")
        )

        mean_decode_time = (
            sum(
                row["decode_time_seconds"]
                for row in group
            )
            / len(group)
        )

        mean_throughput = (
            sum(
                row["shots_per_second"]
                for row in group
            )
            / len(group)
        )

        aggregated.append(
            {
                "run_dir": ";".join(
                    row["run_dir"]
                    for row in group
                ),
                "n_repeats": len(group),
                "det_beam": det_beam,
                "pqlimit": pqlimit,
                "beam_climbing": beam_climbing,
                "distance": distance,
                "physical_error_rate": p_value,
                "logical_failures": total_failures,
                "n_shots": total_shots,
                "logical_error_rate_per_round": (
                    pooled_logical_error_rate
                ),
                "decode_time_seconds": (
                    mean_decode_time
                ),
                "shots_per_second": (
                    mean_throughput
                ),
            }
        )

    aggregated.sort(
        key=lambda row: (
            row["distance"],
            row["physical_error_rate"],
            row["beam_climbing"],
            row["det_beam"],
            row["pqlimit"],
        )
    )

    return aggregated


def _write_summary_csv(
    output_dir: Path,
    rows: list[dict[str, Any]],
) -> Path:
    out = output_dir / "summary.csv"

    fieldnames = [
        "run_dir",
        "n_repeats",
        "det_beam",
        "pqlimit",
        "beam_climbing",
        "distance",
        "physical_error_rate",
        "logical_failures",
        "n_shots",
        "logical_error_rate_per_round",
        "decode_time_seconds",
        "shots_per_second",
    ]

    with out.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    return out


def _write_manifest(
    output_dir: Path,
    runs: list[Beam2DRun],
    args: argparse.Namespace,
    basis: str,
    n_shots: int,
    distances: list[int],
    p_values: list[float],
) -> Path:
    out = output_dir / "manifest.json"

    det_beams = sorted(
        {run.det_beam for run in runs}
    )

    pqlimits = sorted(
        {run.pqlimit for run in runs}
    )

    beam_climbing_values = sorted(
        {run.beam_climbing for run in runs}
    )

    manifest = {
        "timestamp": datetime.now().isoformat(
            timespec="seconds"
        ),
        "plot_script": Path(__file__).name,
        "basis": basis,
        "n_shots": n_shots,
        "distances": distances,
        "p_values": p_values,
        "det_beams": det_beams,
        "pqlimits": pqlimits,
        "beam_climbing_values": beam_climbing_values,
        "run_dirs": [
            str(run.run_dir)
            for run in runs
        ],
        "output_dir": str(output_dir),
    }

    with out.open("w") as f:
        json.dump(
            manifest,
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")

    return out


def _plot_distance_metric(
    *,
    distance: int,
    rows: list[dict[str, Any]],
    metric_key: str,
    metric_label: str,
    metric_slug: str,
    yscale: str,
    output_dir: Path,
    basis: str,
    n_shots: int,
    floor_to_half_shot: bool = False,
) -> Path:
    p_values = sorted(
        {
            row["physical_error_rate"]
            for row in rows
            if row["distance"] == distance
        }
    )

    if not p_values:
        raise ValueError(
            f"No rows found for distance d={distance}"
        )

    colors = list(
        plt.get_cmap("tab10").colors
    )

    color_for_p = {
        p: colors[i % len(colors)]
        for i, p in enumerate(p_values)
    }

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 4.5),
        sharex=True,
        sharey=True,
    )

    beam_states = [
        (False, "Beam climbing OFF"),
        (True, "Beam climbing ON"),
    ]

    floor = (
        half_shot_floor(n_shots)
        if floor_to_half_shot
        else None
    )

    for ax, (
        beam_climbing,
        title_suffix,
    ) in zip(
        axes,
        beam_states,
    ):
        for p in p_values:
            subset = [
                row
                for row in rows
                if row["distance"] == distance
                and row["beam_climbing"]
                == beam_climbing
                and row["physical_error_rate"] == p
            ]

            if not subset:
                continue

            subset.sort(
                key=lambda row: row["det_beam"]
            )

            x = [
                row["det_beam"]
                for row in subset
            ]

            y = [
                row[metric_key]
                for row in subset
            ]

            if floor is not None:
                y = [
                    max(value, floor)
                    for value in y
                ]

            ax.plot(
                x,
                y,
                marker="o",
                color=color_for_p[p],
                label=f"p={p:g}",
            )

        ax.set_title(title_suffix)
        ax.set_xlabel("Detector beam")
        ax.set_ylabel(metric_label)
        ax.set_yscale(yscale)

        ax.grid(
            True,
            which="both",
            linestyle="--",
            alpha=0.4,
        )

        ax.legend(
            title="Physical error rate",
            fontsize=8,
        )

    fig.suptitle(
        f"{basis} | d={distance} | "
        f"{n_shots:,} shots"
    )

    fig.tight_layout(
        rect=[0, 0.02, 1, 0.95]
    )

    out = (
        output_dir
        / (
            f"beam2d_"
            f"{format_shot_count(n_shots)}_"
            f"{basis}_d{distance}_"
            f"{metric_slug}.png"
        )
    )

    save_figure(fig, out)
    plt.close(fig)

    return out


def _plot_overlay_grid(
    *,
    rows: list[dict[str, Any]],
    metric_key: str,
    metric_label: str,
    metric_slug: str,
    yscale: str,
    output_dir: Path,
    basis: str,
    n_shots: int,
    floor_to_half_shot: bool = False,
) -> Path:
    distances = sorted(
        {
            row["distance"]
            for row in rows
        }
    )

    p_values = sorted(
        {
            row["physical_error_rate"]
            for row in rows
        }
    )

    colors = list(
        plt.get_cmap("tab10").colors
    )

    color_for_p = {
        p: colors[i % len(colors)]
        for i, p in enumerate(p_values)
    }

    fig, axes = plt.subplots(
        nrows=len(distances),
        ncols=1,
        figsize=(
            8,
            3.4 * len(distances),
        ),
        sharex=True,
    )

    if len(distances) == 1:
        axes = [axes]

    floor = (
        0.5 / n_shots
        if floor_to_half_shot
        else None
    )

    for ax, distance in zip(
        axes,
        distances,
    ):
        for p in p_values:
            for beam_climbing, linestyle in [
                (False, "-"),
                (True, "--"),
            ]:
                subset = [
                    row
                    for row in rows
                    if row["distance"] == distance
                    and row["physical_error_rate"] == p
                    and row["beam_climbing"]
                    == beam_climbing
                ]

                if not subset:
                    continue

                subset.sort(
                    key=lambda row: row["det_beam"]
                )

                x = [
                    row["det_beam"]
                    for row in subset
                ]

                y = [
                    row[metric_key]
                    for row in subset
                ]

                if floor is not None:
                    y = [
                        max(value, floor)
                        for value in y
                    ]

                label = (
                    f"p={p:g} "
                    + (
                        "ON"
                        if beam_climbing
                        else "OFF"
                    )
                )

                ax.plot(
                    x,
                    y,
                    marker="o",
                    linestyle=linestyle,
                    color=color_for_p[p],
                    label=label,
                )

        ax.set_title(
            f"d = {distance}"
        )

        ax.set_ylabel(
            metric_label
        )

        ax.set_yscale(
            yscale
        )

        ax.grid(
            True,
            which="both",
            linestyle="--",
            alpha=0.4,
        )

    axes[-1].set_xlabel(
        "Detector beam"
    )

    handles, labels = (
        axes[0].get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        fontsize=8,
    )

    fig.suptitle(
        f"{basis} | {n_shots:,} shots",
        fontsize=14,
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    out = (
        output_dir
        / (
            f"beam2d_"
            f"{format_shot_count(n_shots)}_"
            f"{basis}_{metric_slug}_"
            f"overlay.png"
        )
    )

    save_figure(fig, out)
    plt.close(fig)

    return out


def _plot_beam_pq_heatmap(
    *,
    rows: list[dict[str, Any]],
    distance: int,
    p_value: float,
    beam_climbing: bool,
    metric_key: str,
    metric_label: str,
    metric_slug: str,
    output_dir: Path,
    basis: str,
    n_shots: int,
    floor_to_half_shot: bool = False,
) -> Path | None:
    subset = [
        row
        for row in rows
        if row["distance"] == distance
        and row["physical_error_rate"] == p_value
        and row["beam_climbing"] == beam_climbing
    ]

    if not subset:
        return None

    beams = sorted(
        {
            row["det_beam"]
            for row in subset
        }
    )

    pqlimits = sorted(
        {
            row["pqlimit"]
            for row in subset
        }
    )

    if len(beams) < 2 or len(pqlimits) < 2:
        return None

    values: dict[
        tuple[int, float],
        float,
    ] = {}

    for row in subset:
        value = row[metric_key]

        if floor_to_half_shot:
            value = max(
                value,
                half_shot_floor(n_shots),
            )

        values[
            (
                row["pqlimit"],
                row["det_beam"],
            )
        ] = value

    matrix = np.full(
        (
            len(pqlimits),
            len(beams),
        ),
        np.nan,
        dtype=float,
    )

    for row_index, pqlimit in enumerate(
        pqlimits
    ):
        for col_index, beam in enumerate(
            beams
        ):
            value = values.get(
                (
                    pqlimit,
                    beam,
                )
            )

            if value is not None:
                matrix[
                    row_index,
                    col_index,
                ] = value

    fig, ax = plt.subplots(
        figsize=(8.5, 5.5)
    )

    finite_values = matrix[
        np.isfinite(matrix)
    ]

    if finite_values.size == 0:
        plt.close(fig)
        return None

    if metric_key == "logical_error_rate_per_round":
        positive_values = (
            finite_values[
                finite_values > 0
            ]
        )

        if positive_values.size:
            image = ax.imshow(
                matrix,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                cmap="RdYlGn_r",
                norm=LogNorm(
                    vmin=float(
                        positive_values.min()
                    ),
                    vmax=float(
                        positive_values.max()
                    ),
                ),
            )
        else:
            image = ax.imshow(
                matrix,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                cmap="RdYlGn_r",
            )

    elif metric_key == "decode_time_seconds":
        image = ax.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap="magma",
            norm=LogNorm(
                vmin=float(
                    finite_values.min()
                ),
                vmax=float(
                    finite_values.max()
                ),
            ),
        )

    else:
        image = ax.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
        )

    ax.set_xticks(
        range(len(beams))
    )

    ax.set_xticklabels(
        [f"{beam:g}" for beam in beams]
    )

    ax.set_yticks(
        range(len(pqlimits))
    )

    ax.set_yticklabels(
        [f"{pq:,}" for pq in pqlimits]
    )

    ax.set_xlabel(
        "Detector beam"
    )

    ax.set_ylabel(
        "Priority-queue limit"
    )

    ax.set_title(
        f"{basis} | d={distance} | "
        f"p={p_value:g} | "
        f"beam climbing "
        f"{'ON' if beam_climbing else 'OFF'}"
    )

    for row_index, pqlimit in enumerate(
        pqlimits
    ):
        for col_index, beam in enumerate(
            beams
        ):
            value = matrix[
                row_index,
                col_index,
            ]

            if not np.isfinite(value):
                continue

            if metric_key == (
                "logical_error_rate_per_round"
            ):
                text = f"{value:.2e}"

            elif metric_key == (
                "decode_time_seconds"
            ):
                text = f"{value:.0f}"

            else:
                text = f"{value:.1f}"

            ax.text(
                col_index,
                row_index,
                text,
                ha="center",
                va="center",
                fontsize=8,
            )

    colorbar = fig.colorbar(
        image,
        ax=ax,
    )

    colorbar.set_label(
        metric_label
    )

    fig.tight_layout()

    out = (
        output_dir
        / (
            f"beam_pq_heatmap_"
            f"{format_shot_count(n_shots)}_"
            f"{basis}_"
            f"d{distance}_"
            f"p{p_value:g}_"
            f"bc{int(beam_climbing)}_"
            f"{metric_slug}.png"
        )
    )

    save_figure(fig, out)
    plt.close(fig)

    return out


def _plot_bc_ratio_heatmap(
    *,
    rows: list[dict[str, Any]],
    distance: int,
    p_value: float,
    metric_key: str,
    metric_label: str,
    metric_slug: str,
    output_dir: Path,
    basis: str,
    n_shots: int,
) -> Path | None:
    """
    Compare beam-climbing ON against beam-climbing OFF.

    For logical error rate:
        log10(OFF / ON)

    Positive values mean beam climbing improves logical error rate.

    For runtime:
        log10(ON / OFF)

    Positive values mean beam climbing increases runtime.
    """

    subset = [
        row
        for row in rows
        if row["distance"] == distance
        and row["physical_error_rate"] == p_value
    ]

    if not subset:
        return None

    off = {
        (
            row["pqlimit"],
            row["det_beam"],
        ): row[metric_key]
        for row in subset
        if not row["beam_climbing"]
    }

    on = {
        (
            row["pqlimit"],
            row["det_beam"],
        ): row[metric_key]
        for row in subset
        if row["beam_climbing"]
    }

    common_keys = sorted(
        set(off) & set(on)
    )

    if len(common_keys) < 4:
        return None

    beams = sorted(
        {
            beam
            for _, beam in common_keys
        }
    )

    pqlimits = sorted(
        {
            pq
            for pq, _ in common_keys
        }
    )

    matrix = np.full(
        (
            len(pqlimits),
            len(beams),
        ),
        np.nan,
        dtype=float,
    )

    for row_index, pqlimit in enumerate(
        pqlimits
    ):
        for col_index, beam in enumerate(
            beams
        ):
            key = (
                pqlimit,
                beam,
            )

            if key not in off or key not in on:
                continue

            off_value = off[key]
            on_value = on[key]

            if (
                off_value <= 0
                or on_value <= 0
            ):
                continue

            if metric_key == (
                "logical_error_rate_per_round"
            ):
                value = np.log10(
                    off_value / on_value
                )

            elif metric_key == (
                "decode_time_seconds"
            ):
                value = np.log10(
                    on_value / off_value
                )

            else:
                continue

            matrix[
                row_index,
                col_index,
            ] = value

    finite_values = matrix[
        np.isfinite(matrix)
    ]

    if finite_values.size == 0:
        return None

    vmax = float(
        np.max(
            np.abs(
                finite_values
            )
        )
    )

    if vmax == 0:
        vmax = 1.0

    fig, ax = plt.subplots(
        figsize=(8.5, 5.5)
    )

    if metric_key == (
        "logical_error_rate_per_round"
    ):
        cmap = "RdYlGn"
    else:
        cmap = "coolwarm"

    image = ax.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=TwoSlopeNorm(
            vmin=-vmax,
            vcenter=0.0,
            vmax=vmax,
        ),
    )

    ax.set_xticks(
        range(len(beams))
    )

    ax.set_xticklabels(
        [f"{beam:g}" for beam in beams]
    )

    ax.set_yticks(
        range(len(pqlimits))
    )

    ax.set_yticklabels(
        [f"{pq:,}" for pq in pqlimits]
    )

    ax.set_xlabel(
        "Detector beam"
    )

    ax.set_ylabel(
        "Priority-queue limit"
    )

    if metric_key == (
        "logical_error_rate_per_round"
    ):
        title = (
            f"{basis} | d={distance} | "
            f"p={p_value:g} | "
            "beam-climbing benefit"
        )

        colorbar_label = (
            r"$\log_{10}(p_L^{\mathrm{off}}/"
            r"p_L^{\mathrm{on}})$"
        )

    else:
        title = (
            f"{basis} | d={distance} | "
            f"p={p_value:g} | "
            "beam-climbing runtime penalty"
        )

        colorbar_label = (
            r"$\log_{10}(t^{\mathrm{on}}/"
            r"t^{\mathrm{off}})$"
        )

    ax.set_title(title)

    for row_index, pqlimit in enumerate(
        pqlimits
    ):
        for col_index, beam in enumerate(
            beams
        ):
            value = matrix[
                row_index,
                col_index,
            ]

            if not np.isfinite(value):
                continue

            ax.text(
                col_index,
                row_index,
                f"{value:+.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    colorbar = fig.colorbar(
        image,
        ax=ax,
    )

    colorbar.set_label(
        colorbar_label
    )

    fig.tight_layout()

    out = (
        output_dir
        / (
            f"beam_pq_bc_comparison_"
            f"{format_shot_count(n_shots)}_"
            f"{basis}_d{distance}_"
            f"p{p_value:g}_"
            f"{metric_slug}.png"
        )
    )

    save_figure(fig, out)
    plt.close(fig)

    return out


def _print_best_logical_error_summary(
    rows: list[dict[str, Any]],
) -> None:
    print()
    print(
        "best_beam_by_distance,p,"
        "beam_climbing,det_beam,"
        "logical_error_rate_per_round"
    )

    grouped: dict[
        tuple[int, float, bool],
        list[dict[str, Any]],
    ] = {}

    for row in rows:
        key = (
            row["distance"],
            row["physical_error_rate"],
            row["beam_climbing"],
        )

        grouped.setdefault(
            key,
            [],
        ).append(row)

    for (
        distance,
        p_value,
        beam_climbing,
    ), subset in sorted(
        grouped.items()
    ):
        best = min(
            subset,
            key=lambda row: (
                row[
                    "logical_error_rate_per_round"
                ]
            ),
        )

        print(
            f"{distance},"
            f"{p_value:g},"
            f"{int(beam_climbing)},"
            f"{best['det_beam']:g},"
            f"{best['logical_error_rate_per_round']:.6e}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the 2D beam-width / "
            "beam-climbing sweep."
        )
    )

    parser.add_argument(
        "--run-dirs",
        type=Path,
        nargs="+",
        required=True,
        help=(
            "Run directories or experiment "
            "directories to compare."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "experiments/plots/beam2d"
        ),
        help=(
            "Directory to save the "
            "comparison plots into."
        ),
    )

    parser.add_argument(
        "--basis",
        type=str,
        default=None,
        help="Benchmark basis to filter on.",
    )

    parser.add_argument(
        "--decode-mode",
        type=str,
        default=None,
        help="Decode mode to filter on.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
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
        help="Shot count to filter on.",
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
        "--heatmaps",
        action="store_true",
        help=(
            "Also generate detector-beam versus "
            "priority-queue-limit heatmaps."
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
            "No matching runs found."
        )
        return 1

    check_manifests_consistent(
        [
            run.manifest
            for run in runs
        ],
        varying_fields={
            "det_beam",
            "pqlimit",
            "beam_climbing",
            "host",
            "distances",
            "sparsify_base_degree",
            "sparsify_errors",
            "sparsify_max_degree",
            "sparsify_reactivate_limit"
        },
    )

    basis = runs[0].manifest.get(
        "basis",
        "unknown",
    )

    n_shots = int(
        runs[0].manifest["n_shots"]
    )

    raw_rows = _flatten_rows(
        runs
    )

    rows = _aggregate_configuration_rows(
        raw_rows
    )

    repeated_rows = [
        row
        for row in rows
        if row["n_repeats"] > 1
    ]

    if repeated_rows:
        print()
        print(
            "Repeated configurations aggregated:"
        )

        for row in repeated_rows:
            print(
                f"d={row['distance']} "
                f"p={row['physical_error_rate']:g} "
                f"beam={row['det_beam']:g} "
                f"pq={row['pqlimit']} "
                f"bc={int(row['beam_climbing'])} "
                f"repeats={row['n_repeats']}"
            )

    distances = sorted(
        {
            row["distance"]
            for row in rows
        }
    )

    p_values = sorted(
        {
            row[
                "physical_error_rate"
            ]
            for row in rows
        }
    )

    output_dir = args.output_dir

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_csv = _write_summary_csv(
        output_dir,
        rows,
    )

    manifest_path = _write_manifest(
        output_dir=output_dir,
        runs=runs,
        args=args,
        basis=basis,
        n_shots=n_shots,
        distances=distances,
        p_values=p_values,
    )

    saved_paths: list[Path] = []

    metric_order = [
        "logical_error_rate_per_round",
        "decode_time_seconds",
        "shots_per_second",
    ]

    for distance in distances:
        for metric_key in metric_order:
            metric_info = METRICS[
                metric_key
            ]

            saved_paths.append(
                _plot_distance_metric(
                    distance=distance,
                    rows=rows,
                    metric_key=metric_key,
                    metric_label=metric_info[
                        "ylabel"
                    ],
                    metric_slug=metric_info[
                        "slug"
                    ],
                    yscale=metric_info[
                        "yscale"
                    ],
                    output_dir=output_dir,
                    basis=basis,
                    n_shots=n_shots,
                    floor_to_half_shot=bool(
                        metric_info["floor"]
                    ),
                )
            )

    _print_best_logical_error_summary(
        rows
    )

    saved_paths.append(
        _plot_overlay_grid(
            rows=rows,
            metric_key=(
                "logical_error_rate_per_round"
            ),
            metric_label=(
                "Logical error rate per round"
            ),
            metric_slug=(
                "logical_error_rate"
            ),
            yscale="log",
            output_dir=output_dir,
            basis=basis,
            n_shots=n_shots,
            floor_to_half_shot=True,
        )
    )

    saved_paths.append(
        _plot_overlay_grid(
            rows=rows,
            metric_key=(
                "decode_time_seconds"
            ),
            metric_label="Decode time (s)",
            metric_slug="decode_time",
            yscale="linear",
            output_dir=output_dir,
            basis=basis,
            n_shots=n_shots,
        )
    )

    saved_paths.append(
        _plot_overlay_grid(
            rows=rows,
            metric_key="shots_per_second",
            metric_label="Shots per second",
            metric_slug="throughput",
            yscale="linear",
            output_dir=output_dir,
            basis=basis,
            n_shots=n_shots,
        )
    )

    if args.heatmaps:
        beam_climbing_values = sorted(
            {
                row["beam_climbing"]
                for row in rows
            }
        )

        for distance in distances:
            for p_value in p_values:
                for beam_climbing in (
                    beam_climbing_values
                ):
                    for metric_key in (
                        metric_order
                    ):
                        metric_info = METRICS[
                            metric_key
                        ]

                        heatmap_path = (
                            _plot_beam_pq_heatmap(
                                rows=rows,
                                distance=distance,
                                p_value=p_value,
                                beam_climbing=beam_climbing,
                                metric_key=metric_key,
                                metric_label=metric_info[
                                    "ylabel"
                                ],
                                metric_slug=metric_info[
                                    "slug"
                                ],
                                output_dir=output_dir,
                                basis=basis,
                                n_shots=n_shots,
                                floor_to_half_shot=bool(
                                    metric_info[
                                        "floor"
                                    ]
                                ),
                            )
                        )

                        if heatmap_path is not None:
                            saved_paths.append(
                                heatmap_path
                            )

        for distance in distances:
            for p_value in p_values:

                for metric_key in (
                    "logical_error_rate_per_round",
                    "decode_time_seconds",
                ):
                    metric_info = METRICS[
                        metric_key
                    ]

                    path = (
                        _plot_bc_ratio_heatmap(
                            rows=rows,
                            distance=distance,
                            p_value=p_value,
                            metric_key=metric_key,
                            metric_label=metric_info[
                                "ylabel"
                            ],
                            metric_slug=metric_info[
                                "slug"
                            ],
                            output_dir=output_dir,
                            basis=basis,
                            n_shots=n_shots,
                        )
                    )

                    if path is not None:
                        saved_paths.append(
                            path
                        )

    print()
    print(
        f"Saved summary CSV: "
        f"{summary_csv}"
    )

    print(
        f"Saved manifest: "
        f"{manifest_path}"
    )

    for path in saved_paths:
        print(
            f"Saved: {path}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
