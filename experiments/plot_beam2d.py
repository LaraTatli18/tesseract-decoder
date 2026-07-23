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
class Beam2DObservation:
    run_dir: Path
    manifest: dict[str, Any]
    det_beam: float
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


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"Could not parse boolean value from {value!r}")


def _expand_run_dirs(run_dirs: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    seen: set[Path] = set()

    for path in run_dirs:
        if not path.exists():
            print(f"Skipping {path}: does not exist")
            continue

        if (path / "manifest.json").exists() and (path / "results.csv").exists():
            if path not in seen:
                expanded.append(path)
                seen.add(path)
            continue

        for child in sorted(path.iterdir()):
            if not child.is_dir():
                continue
            if (child / "manifest.json").exists() and (child / "results.csv").exists():
                if child not in seen:
                    expanded.append(child)
                    seen.add(child)

    return expanded


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
    if args.distances is not None and [int(v) for v in manifest.get("distances", [])] != args.distances:
        return False
    if args.p_values is not None and [float(v) for v in manifest.get("p_values", [])] != args.p_values:
        return False
    return True


def _collect_observations(run_dirs: list[Path], args: argparse.Namespace) -> list[Beam2DObservation]:
    observations: list[Beam2DObservation] = []

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
        except (KeyError, TypeError, ValueError):
            print(f"Skipping {run_dir}: could not parse 'det_beam' as float")
            continue

        try:
            beam_climbing = _parse_bool(manifest["beam_climbing"])
        except (KeyError, ValueError):
            print(f"Skipping {run_dir}: could not parse 'beam_climbing' as bool")
            continue

        rows = _load_result_rows(results_path)
        if not rows:
            print(f"Skipping {run_dir}: results.csv is empty")
            continue

        observations.append(
            Beam2DObservation(
                run_dir=run_dir,
                manifest=manifest,
                det_beam=det_beam,
                beam_climbing=beam_climbing,
                rows=rows,
            )
        )

    return observations


def _check_manifests_consistent(observations: list[Beam2DObservation]) -> None:
    if not observations:
        return

    reference_run = observations[0]
    reference = reference_run.manifest

    allowed = {
        "timestamp",
        "output_csv",
        "command",
        "det_beam",
        "beam_climbing",
    }

    reference_keys = set(reference.keys()) - allowed

    for obs in observations[1:]:
        manifest = obs.manifest
        for key in reference_keys:
            if manifest.get(key) != reference.get(key):
                raise ValueError(
                    f"Run directory {obs.run_dir} differs in '{key}'.\n"
                    f"Reference run: {reference_run.run_dir}\n"
                    f"Reference value: {reference.get(key)}\n"
                    f"Current value:   {manifest.get(key)}"
                )


def _flatten_rows(observations: list[Beam2DObservation]) -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    for obs in observations:
        for row in obs.rows:
            rows_out.append(
                {
                    "run_dir": str(obs.run_dir),
                    "det_beam": obs.det_beam,
                    "beam_climbing": obs.beam_climbing,
                    "distance": int(row["distance"]),
                    "physical_error_rate": float(row["physical_error_rate"]),
                    "logical_error_rate_per_round": float(row["logical_error_rate_per_round"]),
                    "decode_time_seconds": float(row["decode_time_seconds"]),
                    "shots_per_second": float(row["shots_per_second"]),
                }
            )
    return rows_out


def _write_summary_csv(output_dir: Path, rows: list[dict[str, Any]]) -> Path:
    out = output_dir / "summary.csv"
    fieldnames = [
        "run_dir",
        "det_beam",
        "beam_climbing",
        "distance",
        "physical_error_rate",
        "logical_error_rate_per_round",
        "decode_time_seconds",
        "shots_per_second",
    ]

    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return out


def _write_manifest(
    output_dir: Path,
    observations: list[Beam2DObservation],
    args: argparse.Namespace,
    basis: str,
    n_shots: int,
    distances: list[int],
    p_values: list[float],
) -> Path:
    out = output_dir / "manifest.json"
    det_beams = sorted({obs.det_beam for obs in observations})
    beam_climbing_values = sorted({obs.beam_climbing for obs in observations})

    manifest = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "plot_script": Path(__file__).name,
        "basis": basis,
        "n_shots": n_shots,
        "distances": distances,
        "p_values": p_values,
        "det_beams": det_beams,
        "beam_climbing_values": beam_climbing_values,
        "run_dirs": [str(obs.run_dir) for obs in observations],
        "output_dir": str(output_dir),
    }

    with out.open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
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
    p_values = sorted({row["physical_error_rate"] for row in rows if row["distance"] == distance})
    if not p_values:
        raise ValueError(f"No rows found for distance d={distance}")

    colors = list(plt.get_cmap("tab10").colors)
    color_for_p = {p: colors[i % len(colors)] for i, p in enumerate(p_values)}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharex=True, sharey=True)
    beam_states = [(False, "Beam climbing OFF"), (True, "Beam climbing ON")]

    floor = 0.5 / n_shots if floor_to_half_shot else None

    for ax, (beam_climbing, title_suffix) in zip(axes, beam_states):
        for p in p_values:
            subset = [
                row
                for row in rows
                if row["distance"] == distance
                and row["beam_climbing"] == beam_climbing
                and row["physical_error_rate"] == p
            ]
            if not subset:
                continue

            subset.sort(key=lambda row: row["det_beam"])
            x = [row["det_beam"] for row in subset]
            y = [row[metric_key] for row in subset]
            if floor is not None:
                y = [max(value, floor) for value in y]

            ax.plot(x, y, marker="o", color=color_for_p[p], label=f"p={p:g}")

        ax.set_title(title_suffix)
        ax.set_xlabel("Detector beam")
        ax.set_ylabel(metric_label)
        ax.set_yscale(yscale)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend(title="Physical error rate", fontsize=8)

    fig.suptitle(f"{basis} | d={distance} | {n_shots:,} shots")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    out = output_dir / f"beam2d_{_format_shot_count(n_shots)}_{basis}_d{distance}_{metric_slug}.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out

def _plot_overlay_grid(
    *,
    rows,
    metric_key,
    metric_label,
    metric_slug,
    yscale,
    output_dir,
    basis,
    n_shots,
    floor_to_half_shot=False,
):
    distances = sorted({row["distance"] for row in rows})
    p_values = sorted({row["physical_error_rate"] for row in rows})

    colors = list(plt.get_cmap("tab10").colors)
    color_for_p = {
        p: colors[i % len(colors)]
        for i, p in enumerate(p_values)
    }

    fig, axes = plt.subplots(
        nrows=len(distances),
        ncols=1,
        figsize=(8, 3.4 * len(distances)),
        sharex=True,
    )

    if len(distances) == 1:
        axes = [axes]

    floor = 0.5 / n_shots if floor_to_half_shot else None

    for ax, distance in zip(axes, distances):

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
                    and row["beam_climbing"] == beam_climbing
                ]

                if not subset:
                    continue

                subset.sort(key=lambda r: r["det_beam"])

                x = [r["det_beam"] for r in subset]
                y = [r[metric_key] for r in subset]

                if floor is not None:
                    y = [max(v, floor) for v in y]

                label = (
                    f"p={p:g} "
                    + ("ON" if beam_climbing else "OFF")
                )

                ax.plot(
                    x,
                    y,
                    marker="o",
                    linestyle=linestyle,
                    color=color_for_p[p],
                    label=label,
                )

        ax.set_title(f"d = {distance}")
        ax.set_ylabel(metric_label)
        ax.set_yscale(yscale)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("Detector beam")

    handles, labels = axes[0].get_legend_handles_labels()

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

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out = (
        output_dir
        / f"beam2d_{_format_shot_count(n_shots)}_{basis}_{metric_slug}_overlay.png"
    )

    fig.savefig(out, dpi=300)
    plt.close(fig)

    return out


def _print_best_logical_error_summary(rows: list[dict[str, Any]]) -> None:
    print()
    print("best_beam_by_distance,p,beam_climbing,det_beam,logical_error_rate_per_round")
    grouped: dict[tuple[int, float, bool], list[dict[str, Any]]] = {}

    for row in rows:
        key = (row["distance"], row["physical_error_rate"], row["beam_climbing"])
        grouped.setdefault(key, []).append(row)

    for (distance, p_value, beam_climbing), subset in sorted(grouped.items()):
        best = min(subset, key=lambda row: row["logical_error_rate_per_round"])
        print(
            f"{distance},{p_value:g},{int(beam_climbing)},{best['det_beam']:g},"
            f"{best['logical_error_rate_per_round']:.6e}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot the 2D beam-width / beam-climbing sweep.")
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
        default=Path("experiments/plots/beam2d"),
        help="Directory to save the comparison plots into.",
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
        help="Circuit worker count to filter on.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
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

    run_dirs = _expand_run_dirs(args.run_dirs)
    if not run_dirs:
        print("No run directories found.")
        return 1

    observations = _collect_observations(run_dirs, args)
    if not observations:
        print("No matching runs found.")
        return 1

    _check_manifests_consistent(observations)

    basis = observations[0].manifest.get("basis", "unknown")
    n_shots = int(observations[0].manifest["n_shots"])

    rows = _flatten_rows(observations)
    distances = sorted({row["distance"] for row in rows})
    p_values = sorted({row["physical_error_rate"] for row in rows})

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = _write_summary_csv(output_dir, rows)
    manifest_path = _write_manifest(
        output_dir=output_dir,
        observations=observations,
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
            metric_info = METRICS[metric_key]
            saved_paths.append(
                _plot_distance_metric(
                    distance=distance,
                    rows=rows,
                    metric_key=metric_key,
                    metric_label=metric_info["ylabel"],
                    metric_slug=metric_info["slug"],
                    yscale=metric_info["yscale"],
                    output_dir=output_dir,
                    basis=basis,
                    n_shots=n_shots,
                    floor_to_half_shot=bool(metric_info["floor"]),
                )
            )

    _print_best_logical_error_summary(rows)
    saved_paths.append(
        _plot_overlay_grid(
            rows=rows,
            metric_key="logical_error_rate_per_round",
            metric_label="Logical error rate per round",
            metric_slug="logical_error_rate",
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
            metric_key="decode_time_seconds",
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

    print()
    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved manifest: {manifest_path}")
    for path in saved_paths:
        print(f"Saved: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
