from __future__ import annotations

from collections import defaultdict

import numpy as np
import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

@dataclass(frozen=True)
class PQLimitSweepRecord:
    distance: int
    physical_error_rate: float
    det_beam: float
    pqlimit: float
    logical_error_rate_per_round: float
    decode_time_seconds: float
    shots_per_second: float
    n_shots: int

def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    with manifest_path.open() as f:
        return json.load(f)


def _load_result_rows(results_path: Path) -> list[dict[str, str]]:
    with results_path.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _as_float_list(values: list[Any]) -> list[float]:
    return [float(v) for v in values]

def _expand_run_dirs(run_dirs: list[Path]) -> list[Path]:
    expanded: list[Path] = []

    for path in run_dirs:
        if not path.exists():
            print(f"Skipping {path}: does not exist")
            continue

        if (path / "manifest.json").exists() and (path / "results.csv").exists():
            expanded.append(path)
            continue

        for child in sorted(path.iterdir()):
            if not child.is_dir():
                continue
            if (child / "manifest.json").exists() and (child / "results.csv").exists():
                expanded.append(child)

    return expanded


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalised = value.strip().lower()
        if normalised in {"1", "true", "yes", "on"}:
            return True
        if normalised in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"Could not parse boolean value from {value!r}")


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
    if args.beam_climbing is not None and _parse_bool(manifest.get("beam_climbing", False)) != args.beam_climbing:
        return False
    if args.merge_errors is not None and _parse_bool(manifest.get("merge_errors", False)) != args.merge_errors:
        return False
    if args.distances is not None:
        manifest_distances = sorted(int(v) for v in manifest.get("distances", []))
        if manifest_distances != sorted(args.distances):
            return False
    if args.p_values is not None:
        manifest_p_values = sorted(float(v) for v in manifest.get("p_values", []))
        if manifest_p_values != sorted(args.p_values):
            return False
    return True


def _collect_records(run_dirs: list[Path], args: argparse.Namespace) -> list[PQLimitSweepRecord]:
    records: list[PQLimitSweepRecord] = []

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

        try:
            det_beam = float(manifest["det_beam"])
            pqlimit = float(manifest["pqlimit"])
        except (KeyError, TypeError, ValueError):
            print(f"Skipping {run_dir}: missing or invalid det_beam/pqlimit in manifest")
            continue

        rows = _load_result_rows(results_path)
        if not rows:
            print(f"Skipping {run_dir}: results.csv is empty")
            continue

        for row in rows:
            try:
                records.append(
                    PQLimitSweepRecord(
                        distance=int(row["distance"]),
                        physical_error_rate=float(row["physical_error_rate"]),
                        det_beam=det_beam,
                        pqlimit=pqlimit,
                        logical_error_rate_per_round=float(row["logical_error_rate_per_round"]),
                        decode_time_seconds=float(row["decode_time_seconds"]),
                        shots_per_second=float(row["shots_per_second"]),
                        n_shots=int(row["n_shots"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                print(f"Skipping malformed row in {results_path}")
                continue

    return records

def _aggregate_records(
    records: list[PQLimitSweepRecord],
) -> dict[tuple[int, float, float, float], dict[str, float]]:
    grouped: dict[tuple[int, float, float, float], list[PQLimitSweepRecord]] = defaultdict(list)

    for record in records:
        key = (record.distance, record.physical_error_rate, record.det_beam, record.pqlimit)
        grouped[key].append(record)

    aggregated: dict[tuple[int, float, float, float], dict[str, float]] = {}
    for key, items in grouped.items():
        aggregated[key] = {
            "logical_error_rate_per_round": float(np.mean([r.logical_error_rate_per_round for r in items])),
            "decode_time_seconds": float(np.mean([r.decode_time_seconds for r in items])),
            "shots_per_second": float(np.mean([r.shots_per_second for r in items])),
            "n_shots": float(items[0].n_shots),
        }

    return aggregated


def _format_p_value(p_value: float) -> str:
    return f"p{p_value:g}".replace(".", "p")


def _log_floor(n_shots: int) -> float:
    return np.log10(0.5 / max(n_shots, 1))


def _geometric_edges(values: list[float]) -> np.ndarray:
    vals = sorted(values)
    if len(vals) == 1:
        v = vals[0]
        return np.array([v / np.sqrt(10.0), v * np.sqrt(10.0)])

    edges = [vals[0] / np.sqrt(vals[1] / vals[0])]
    for i in range(1, len(vals)):
        edges.append(np.sqrt(vals[i - 1] * vals[i]))
    edges.append(vals[-1] * np.sqrt(vals[-1] / vals[-2]))
    return np.array(edges)


def _linear_edges(values: list[float]) -> np.ndarray:
    vals = sorted(values)
    if len(vals) == 1:
        v = vals[0]
        return np.array([v - 0.5, v + 0.5])

    edges = [vals[0] - (vals[1] - vals[0]) / 2]
    for i in range(1, len(vals)):
        edges.append((vals[i - 1] + vals[i]) / 2)
    edges.append(vals[-1] + (vals[-1] - vals[-2]) / 2)
    return np.array(edges)

def _compute_global_heatmap_scale(
    aggregated: dict[tuple[int, float, float, float], dict[str, float]],
) -> tuple[float, float]:
    values: list[float] = []

    for stats in aggregated.values():
        n_shots = int(stats["n_shots"])
        floor = 0.5 / max(n_shots, 1)
        v = max(stats["logical_error_rate_per_round"], floor)
        values.append(np.log10(v))

    if not values:
        raise ValueError("No data available to compute heatmap scale")

    vmin = min(values)
    vmax = max(values)
    if vmin == vmax:
        vmin -= 1e-12
        vmax += 1e-12

    return vmin, vmax

def _plot_line_cross_sections(
    aggregated: dict[tuple[int, float, float, float], dict[str, float]],
    *,
    p_value: float,
    basis: str,
    output_dir: Path,
) -> Path:
    distances = sorted({d for (d, p, _, _) in aggregated.keys() if p == p_value})
    beams = sorted({beam for (d, p, beam, _) in aggregated.keys() if p == p_value})
    pqlimits = sorted({pq for (d, p, _, pq) in aggregated.keys() if p == p_value})

    if not distances:
        raise ValueError(f"No data found for p={p_value:g}")

    fig, axes = plt.subplots(
        nrows=len(distances),
        ncols=1,
        figsize=(9.5, 2.5 * len(distances)),
        sharex=True,
        sharey=True,
    )

    if len(distances) == 1:
        axes = [axes]

    colours = list(plt.get_cmap("tab10").colors)
    colour_for_beam = {beam: colours[i % len(colours)] for i, beam in enumerate(beams)}

    n_shots = int(next(iter(aggregated.values()))["n_shots"])
    floor = 0.5 / n_shots

    for ax, distance in zip(axes, distances):
        for beam in beams:
            xs: list[float] = []
            ys: list[float] = []

            for pq in pqlimits:
                key = (distance, p_value, beam, pq)
                if key not in aggregated:
                    continue
                xs.append(pq)
                ys.append(max(aggregated[key]["logical_error_rate_per_round"], floor))

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
        ax.set_yscale("log")
        ax.minorticks_on()
        ax.grid(True, which="major", linestyle="--", alpha=0.3)
        ax.grid(True, which="minor", linestyle=":", alpha=0.15)
        ax.set_title(f"d = {distance}", fontsize=12)

    axes[-1].set_xlabel("Priority queue limit")
    axes[0].set_ylabel("Logical error rate per round")

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
        f"{basis} – Priority queue sweep (p = {p_value:g})",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out = output_dir / f"{basis}_{_format_p_value(p_value)}_line_cross_sections.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out

def _plot_heatmaps(
    aggregated: dict[tuple[int, float, float, float], dict[str, float]],
    *,
    p_value: float,
    basis: str,
    output_dir: Path,
    abs_vmin: float,
    abs_vmax: float,
    relative: bool = False,
) -> Path:

    distances = sorted({d for (d, p, _, _) in aggregated.keys() if p == p_value})
    beams = sorted({beam for (d, p, beam, _) in aggregated.keys() if p == p_value})
    pqlimits = sorted({pq for (d, p, _, pq) in aggregated.keys() if p == p_value})

    if not distances:
        raise ValueError(f"No data found for p={p_value:g}")

    fig, axes = plt.subplots(
        nrows=len(distances),
        ncols=1,
        figsize=(8.5, 2.8 * len(distances)),
        sharex=True,
    )
    if len(distances) == 1:
        axes = [axes]

    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("lightgrey")

    x_edges = _geometric_edges(pqlimits)
    y_edges = _linear_edges(beams)
    last_mesh = None

    for ax, distance in zip(axes, distances):
        matrix = np.full((len(beams), len(pqlimits)), np.nan, dtype=float)

        values = []

        for beam in beams:
            for pq in pqlimits:
                key = (distance, p_value, beam, pq)
                if key not in aggregated:
                    continue

                stats = aggregated[key]
                floor = 0.5 / max(int(stats["n_shots"]), 1)
                values.append(
                    max(
                        stats["logical_error_rate_per_round"],
                        floor,
                    )
                )

        best = min(values)

        for i, beam in enumerate(beams):


            for j, pq in enumerate(pqlimits):
                key = (distance, p_value, beam, pq)
                if key not in aggregated:
                    continue
                stats = aggregated[key]
                floor = 0.5 / max(int(stats["n_shots"]), 1)

                v = max(
                    stats["logical_error_rate_per_round"],
                    floor,
                )
                if relative:
                    matrix[i, j] = np.log10(v / best)
                else:
                    matrix[i, j] = np.log10(v)

        masked = np.ma.masked_invalid(matrix)
        if relative:
            last_mesh = ax.pcolormesh(
                x_edges,
                y_edges,
                masked,
                shading="auto",
                cmap=cmap,
                vmin=0,
                vmax=np.nanmax(masked),
            )
        else:
            last_mesh = ax.pcolormesh(
                x_edges,
                y_edges,
                masked,
                shading="auto",
                cmap=cmap,
                vmin=abs_vmin,
                vmax=abs_vmax,
            )
        ax.set_xscale("log")
        ax.set_ylabel("det_beam")
        ax.set_yticks(beams)
        ax.set_yticklabels([f"{beam:g}" for beam in beams])
        ax.set_title(f"d={distance}, p={p_value:g}", fontsize=11)
        ax.grid(False)

    axes[-1].set_xlabel("Priority queue limit")
    fig.subplots_adjust(right=0.86)

    cbar = fig.colorbar(
        last_mesh,
        ax=axes,
        fraction=0.04,
        pad=0.02,
    )

    if relative:
        cbar.set_label(
            r"$\log_{10}(\mathrm{LER}/\mathrm{Best\ LER})$"
        )
    else:
        cbar.set_label(
            r"$\log_{10}(\mathrm{Logical\ Error\ Rate\ per\ Round})$"
        )

    fig.suptitle(f"{basis} | p={p_value:g} | heatmaps", y=0.995)

    suffix = "relative_heatmaps" if relative else "heatmaps"

    out = output_dir / (
        f"{basis}_{_format_p_value(p_value)}_{suffix}.png"
    )
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out

def main() -> int:
    parser = argparse.ArgumentParser(description="Plot the pqlimit sweep.")
    parser.add_argument(
        "--run-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="Run directories or experiment directories to compare.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/plots/pqlimit_sweep"),
        help="Directory to save the plots into.",
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
    parser.add_argument(
        "--beam-climbing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter beam climbing on/off.",
    )
    parser.add_argument(
        "--merge-errors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter merge_errors on/off.",
    )
    args = parser.parse_args()

    run_dirs = _expand_run_dirs(args.run_dirs)
    if not run_dirs:
        print("No run directories found.")
        return 1

    records = _collect_records(run_dirs, args)
    if not records:
        print("No matching rows found.")
        return 1

    aggregated = _aggregate_records(records)
    abs_vmin, abs_vmax = _compute_global_heatmap_scale(aggregated)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    p_values = sorted({record.physical_error_rate for record in records})
    print("p_value,figure_type,output_file")
    for p_value in p_values:
        line_png = _plot_line_cross_sections(
            aggregated,
            p_value=p_value,
            basis=args.basis,
            output_dir=output_dir,
        )
        heatmap_png = _plot_heatmaps(
            aggregated,
            p_value=p_value,
            basis=args.basis,
            output_dir=output_dir,
            abs_vmin=abs_vmin,
            abs_vmax=abs_vmax,
            relative=False,
        )

        relative_png = _plot_heatmaps(
            aggregated,
            p_value=p_value,
            basis=args.basis,
            output_dir=output_dir,
            abs_vmin=abs_vmin,
            abs_vmax=abs_vmax,
            relative=True,
        )
        print(f"{p_value:g},line,{line_png}")
        print(f"{p_value:g},heatmap,{heatmap_png}")
        print(f"{p_value:g},relative_heatmap,{relative_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

