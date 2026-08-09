from __future__ import annotations

"""Plot sparsification sweeps across distance for fixed beam and pqlimit settings.

This script reads benchmark run directories, extracts one-row summaries from each run,
and plots logical error rate per round, decode time, and throughput versus code
distance for sparsify on/off at each (p_value, det_beam, pqlimit) combination.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from plot_utils import (
    expand_run_dirs,
    load_manifest,
    load_rows,
    save_figure,
    setup_matplotlib,
    format_p_value,
    safe_filename_component
)


@dataclass(frozen=True)
class RunSummary:
    run_dir: Path
    basis: str
    distance: int
    p_value: float
    det_beam: int
    pqlimit: int
    sparsify_errors: bool
    decode_time_seconds: float
    shots_per_second: float
    logical_error_rate_per_round: float
    logical_error_rate_per_round_ci_low: float
    logical_error_rate_per_round_ci_high: float


@dataclass(frozen=True)
class MetricSpec:
    attr: str
    ylabel: str
    title_fragment: str
    yscale: str = "linear"
    yscale_kwargs: dict[str, Any] | None = None
    use_errorbars: bool = False


METRICS: dict[str, MetricSpec] = {
    "logical_error_rate_per_round": MetricSpec(
        attr="logical_error_rate_per_round",
        ylabel="Logical error rate / round",
        title_fragment="decoder quality",
        yscale="symlog",
        yscale_kwargs={"linthresh": 1e-7},
        use_errorbars=True,
    ),
    "decode_time_seconds": MetricSpec(
        attr="decode_time_seconds",
        ylabel="Decode time (s)",
        title_fragment="runtime",
    ),
    "shots_per_second": MetricSpec(
        attr="shots_per_second",
        ylabel="Shots per second",
        title_fragment="throughput",
    ),
}


def _as_single_value(value: Any, *, key: str) -> Any:
    """
    Manifest values like distances and p_values are stored as lists.
    For this sweep we require exactly one value in each run.
    """
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(
                f"Expected manifest[{key!r}] to contain exactly one value, got {value!r}"
            )
        return value[0]
    return value


def _load_summary(run_dir: Path) -> RunSummary | None:
    manifest_path = run_dir / "manifest.json"
    results_path = run_dir / "results.csv"

    if not manifest_path.exists() or not results_path.exists():
        return None

    manifest = load_manifest(manifest_path)
    rows = load_rows(results_path)
    if not rows:
        return None

    if "sparsify_errors" not in manifest:
        return None

    basis = str(manifest["basis"])
    sparsify_errors = bool(manifest["sparsify_errors"])

    distance = int(_as_single_value(manifest["distances"], key="distances"))
    p_value = float(_as_single_value(manifest["p_values"], key="p_values"))
    det_beam = int(manifest["det_beam"])
    pqlimit = int(manifest["pqlimit"])

    row = rows[0]
    decode_time_seconds = float(row["decode_time_seconds"])
    shots_per_second = float(row["shots_per_second"])
    logical_error_rate_per_round = float(row["logical_error_rate_per_round"])

    rounds = int(row["rounds"])
    if "logical_error_rate_ci_low" in row and "logical_error_rate_ci_high" in row:
        low = float(row["logical_error_rate_ci_low"])
        high = float(row["logical_error_rate_ci_high"])
        logical_error_rate_per_round_ci_low = 0.5 * (1 - (1 - 2 * low) ** (1 / rounds))
        logical_error_rate_per_round_ci_high = 0.5 * (1 - (1 - 2 * high) ** (1 / rounds))
    else:
        logical_error_rate_per_round_ci_low = logical_error_rate_per_round
        logical_error_rate_per_round_ci_high = logical_error_rate_per_round

    return RunSummary(
        run_dir=run_dir,
        basis=basis,
        distance=distance,
        p_value=p_value,
        det_beam=det_beam,
        pqlimit=pqlimit,
        sparsify_errors=sparsify_errors,
        decode_time_seconds=decode_time_seconds,
        shots_per_second=shots_per_second,
        logical_error_rate_per_round=logical_error_rate_per_round,
        logical_error_rate_per_round_ci_low=logical_error_rate_per_round_ci_low,
        logical_error_rate_per_round_ci_high=logical_error_rate_per_round_ci_high,
    )

def _group_runs(
    summaries: list[RunSummary],
) -> dict[tuple[float, int, int], dict[bool, list[RunSummary]]]:
    """
    Group runs as:
      (p_value, det_beam, pqlimit) -> sparsify_errors -> list[RunSummary]
    """
    grouped: dict[tuple[float, int, int], dict[bool, list[RunSummary]]] = {}
    for summary in summaries:
        key = (summary.p_value, summary.det_beam, summary.pqlimit)
        grouped.setdefault(key, {}).setdefault(summary.sparsify_errors, []).append(summary)
    return grouped


def _plot_one_metric(
    metric_name: str,
    grouped: dict[tuple[float, int, int], dict[bool, list[RunSummary]]],
    output_dir: Path,
    basis: str,
) -> list[Path]:
    spec = METRICS[metric_name]
    saved_paths: list[Path] = []

    for (p_value, det_beam, pqlimit), by_sparsify in sorted(grouped.items(), key=lambda item: item[0]):
        fig, ax = plt.subplots(figsize=(7.5, 4.5))

        for sparsify_errors, runs in sorted(by_sparsify.items(), key=lambda item: item[0]):
            runs = sorted(runs, key=lambda r: r.distance)

            x = [r.distance for r in runs]
            y = [getattr(r, spec.attr) for r in runs]

            label = "sparsify on" if sparsify_errors else "sparsify off"

            if spec.use_errorbars:
                yerr_low = [
                    max(0.0, getattr(r, spec.attr) - r.logical_error_rate_per_round_ci_low)
                    for r in runs
                ]
                yerr_high = [
                    max(0.0, r.logical_error_rate_per_round_ci_high - getattr(r, spec.attr))
                    for r in runs
                ]
                ax.errorbar(x, y, yerr=[yerr_low, yerr_high], marker="o", capsize=4, label=label)
            else:
                ax.plot(x, y, marker="o", label=label)

        ax.set_xlabel("Distance")
        ax.set_ylabel(spec.ylabel)
        ax.set_title(
            f"{basis} sparsification sweep: {spec.title_fragment} "
            f"p={p_value:g}, beam={det_beam}, pqlimit={pqlimit}"
        )
        ax.set_yscale(spec.yscale, **(spec.yscale_kwargs or {}))
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend()

        fig.tight_layout()

        out = output_dir / (
            f"{safe_filename_component(basis)}_"
            f"{format_p_value(p_value)}_"
            f"beam{det_beam}_"
            f"pq{pqlimit}_"
            f"{metric_name}.png"
        )
        save_figure(fig, out)
        plt.close(fig)
        saved_paths.append(out)

    return saved_paths


def main() -> int:
    setup_matplotlib()

    parser = argparse.ArgumentParser(description="Plot the sparsification sweep.")
    parser.add_argument(
        "--run-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="One or more run directories, or parent directories containing run directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/plots/sparsification"),
        help="Directory to save plots into.",
    )
    parser.add_argument(
        "--metric",
        action="append",
        choices=list(METRICS),
        default=None,
        help="Metric to plot. Repeat to make multiple plots. Defaults to all metrics.",
    )
    parser.add_argument(
        "--basis",
        type=str,
        default=None,
        help="Optional basis filter, e.g. surface_code_X.",
    )
    args = parser.parse_args()

    run_dirs = expand_run_dirs(args.run_dirs)
    summaries: list[RunSummary] = []

    for run_dir in run_dirs:
        summary = _load_summary(run_dir)
        if summary is None:
            continue
        if args.basis is not None and summary.basis != args.basis:
            continue
        summaries.append(summary)

    if not summaries:
        print("No matching runs found.")
        return 1

    reference = summaries[0]
    for s in summaries[1:]:
        if s.basis != reference.basis:
            raise ValueError(f"Mixed basis values found: {reference.basis!r} vs {s.basis!r}")

    grouped = _group_runs(summaries)
    selected_metrics = args.metric if args.metric is not None else list(METRICS)

    saved: list[Path] = []
    for metric_name in selected_metrics:
        saved.extend(_plot_one_metric(metric_name, grouped, args.output_dir, reference.basis))

    print("Saved plots:")
    for path in saved:
        print(f"  {path}")

    print()
    print("Summary table:")
    print("p_value,det_beam,pqlimit,distance,sparsify_errors,logical_error_rate_per_round,decode_time_seconds,shots_per_second")
    for (p_value, det_beam, pqlimit) in sorted(grouped):
        for sparsify_errors in sorted(grouped[(p_value, det_beam, pqlimit)]):
            for s in sorted(grouped[(p_value, det_beam, pqlimit)][sparsify_errors], key=lambda r: r.distance):
                print(
                    f"{s.p_value:g},"
                    f"{s.det_beam},"
                    f"{s.pqlimit},"
                    f"{s.distance},"
                    f"{int(s.sparsify_errors)},"
                    f"{s.logical_error_rate_per_round:.8e},"
                    f"{s.decode_time_seconds:.6f},"
                    f"{s.shots_per_second:.2f}"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
