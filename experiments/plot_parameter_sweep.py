from __future__ import annotations

"""Plot generic one- or two-parameter Tesseract decoder sweeps.

This script loads benchmark runs and groups them by an arbitrary manifest parameter,
optionally splitting each sweep by a secondary parameter. It compares decoder
runtime, throughput, and logical error rate per round, including confidence
intervals for decoder-quality plots.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

import matplotlib.pyplot as plt

from plot_utils import (
    check_manifests_consistent,
    collect_plot_runs,
    expand_run_dirs,
    format_number,
    matches_filters,
    safe_filename_component,
    save_figure,
    setup_matplotlib,
)


PARAMETER_ALIASES: dict[str, str] = {
    "distance": "distances",
    "p": "p_values",
    "beam": "det_beam",
    "pqlimit": "pqlimit",
}


@dataclass(frozen=True)
class NormalisedValue:
    key: Any
    label: str
    axis_value: float | None
    sort_key: tuple[Any, ...]


@dataclass(frozen=True)
class ParameterSweepRun:
    normalised_value: NormalisedValue
    mean_decode_time_seconds: float
    mean_shots_per_second: float
    mean_logical_error_rate_per_round: float
    mean_logical_error_rate_per_round_ci_low: float
    mean_logical_error_rate_per_round_ci_high: float
    run_dirs: list[Path]
    manifests: list[dict[str, Any]]


@dataclass(frozen=True)
class MetricSpec:
    attribute: str
    ylabel: str
    title_fragment: str
    yscale: str = "linear"
    yscale_kwargs: dict[str, Any] | None = None
    use_errorbars: bool = False


METRICS: dict[str, MetricSpec] = {
    "decode_time_seconds": MetricSpec(
        attribute="mean_decode_time_seconds",
        ylabel="Decode time (s)",
        title_fragment="decode time",
    ),
    "shots_per_second": MetricSpec(
        attribute="mean_shots_per_second",
        ylabel="Shots per second",
        title_fragment="throughput",
    ),
    "logical_error_rate_per_round": MetricSpec(
        attribute="mean_logical_error_rate_per_round",
        ylabel="Logical error rate / round",
        title_fragment="decoder quality",
        yscale="symlog",
        yscale_kwargs={"linthresh": 1e-7},
        use_errorbars=True,
    ),
}


def _normalise_manifest_value(value: Any) -> NormalisedValue:
    if isinstance(value, bool):
        return NormalisedValue(
            key=("bool", value),
            label="on" if value else "off",
            axis_value=1.0 if value else 0.0,
            sort_key=(0, int(value)),
        )

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        return NormalisedValue(
            key=("num", numeric),
            label=format_number(numeric),
            axis_value=numeric,
            sort_key=(1, numeric),
        )

    if isinstance(value, (list, tuple)):
        normalised_items = [_normalise_manifest_value(item) for item in value]
        if len(normalised_items) == 1:
            item = normalised_items[0]
            return NormalisedValue(
                key=item.key,
                label=item.label,
                axis_value=item.axis_value,
                sort_key=(2,) + item.sort_key,
            )

        return NormalisedValue(
            key=("list", tuple(item.key for item in normalised_items)),
            label="[" + ", ".join(item.label for item in normalised_items) + "]",
            axis_value=None,
            sort_key=(3, tuple(item.sort_key for item in normalised_items)),
        )

    if value is None:
        return NormalisedValue(
            key=("none", None),
            label="None",
            axis_value=None,
            sort_key=(4, 0),
        )

    text = str(value)
    return NormalisedValue(
        key=("str", text),
        label=text,
        axis_value=None,
        sort_key=(5, text),
    )


def _resolve_parameter_name(parameter: str, manifest: dict[str, Any]) -> str:
    if parameter in manifest:
        return parameter
    alias = PARAMETER_ALIASES.get(parameter)
    if alias is not None and alias in manifest:
        return alias
    return parameter


def _group_runs_by_parameter(
    run_dirs: list[Path],
    args: argparse.Namespace,
) -> tuple[
    dict[
        tuple[Any, Any],
        list[
            tuple[
                Path,
                dict[str, Any],
                list[dict[str, str]],
                NormalisedValue,
                NormalisedValue | None,
            ]
        ],
    ],
    list[tuple[Path, dict[str, Any]]],
    str,
]:
    grouped: dict[
        tuple[Any, Any],
        list[
            tuple[
                Path,
                dict[str, Any],
                list[dict[str, str]],
                NormalisedValue,
                NormalisedValue | None,
            ]
        ],
    ] = {}

    run_records: list[tuple[Path, dict[str, Any]]] = []
    resolved_parameter_name = args.parameter

    for run in collect_plot_runs(run_dirs):
        manifest = run.manifest

        if not matches_filters(manifest, args):
            continue

        manifest_key = _resolve_parameter_name(args.parameter, manifest)

        if manifest_key not in manifest:
            print(f"Skipping {run.run_dir}: manifest missing '{args.parameter}'")
            continue

        resolved_parameter_name = manifest_key

        try:
            normalised_value = _normalise_manifest_value(manifest[manifest_key])
        except (TypeError, ValueError):
            print(f"Skipping {run.run_dir}: could not normalise '{manifest_key}'")
            continue

        secondary_normalised_value: NormalisedValue | None = None

        if args.secondary_parameter is not None:
            secondary_key = _resolve_parameter_name(args.secondary_parameter, manifest)

            if secondary_key not in manifest:
                print(
                    f"Skipping {run.run_dir}: manifest missing '{args.secondary_parameter}'"
                )
                continue

            try:
                secondary_normalised_value = _normalise_manifest_value(manifest[secondary_key])
            except (TypeError, ValueError):
                print(f"Skipping {run.run_dir}: could not normalise '{secondary_key}'")
                continue

        group_key = (
            normalised_value.key,
            secondary_normalised_value.key if secondary_normalised_value is not None else None,
        )

        grouped.setdefault(group_key, []).append(
            (
                run.run_dir,
                manifest,
                run.rows,
                normalised_value,
                secondary_normalised_value,
            )
        )

        run_records.append((run.run_dir, manifest))

    return grouped, run_records, resolved_parameter_name


def _summarise_group(
    normalised_value: NormalisedValue,
    runs: list[
        tuple[
            Path,
            dict[str, Any],
            list[dict[str, str]],
            NormalisedValue,
            NormalisedValue | None,
        ]
    ],
) -> ParameterSweepRun:
    decode_time_values: list[float] = []
    throughput_values: list[float] = []
    quality_values: list[float] = []
    quality_ci_low_values: list[float] = []
    quality_ci_high_values: list[float] = []
    run_dirs: list[Path] = []
    manifests: list[dict[str, Any]] = []

    for run_dir, manifest, rows, _, _secondary in runs:
        run_dirs.append(run_dir)
        manifests.append(manifest)

        for row in rows:
            rounds = float(row["rounds"])
            decode_time_values.append(float(row["decode_time_seconds"]))
            throughput_values.append(float(row["shots_per_second"]))
            quality_values.append(float(row["logical_error_rate_per_round"]))

            if "logical_error_rate_ci_low" in row and "logical_error_rate_ci_high" in row:
                quality_ci_low_values.append(float(row["logical_error_rate_ci_low"]) / rounds)
                quality_ci_high_values.append(float(row["logical_error_rate_ci_high"]) / rounds)
            else:
                quality_ci_low_values.append(float(row["logical_error_rate_per_round"]))
                quality_ci_high_values.append(float(row["logical_error_rate_per_round"]))

    return ParameterSweepRun(
        normalised_value=normalised_value,
        mean_decode_time_seconds=fmean(decode_time_values),
        mean_shots_per_second=fmean(throughput_values),
        mean_logical_error_rate_per_round=fmean(quality_values),
        mean_logical_error_rate_per_round_ci_low=fmean(quality_ci_low_values),
        mean_logical_error_rate_per_round_ci_high=fmean(quality_ci_high_values),
        run_dirs=run_dirs,
        manifests=manifests,
    )


def _shared_manifest_value(points: list[ParameterSweepRun], key: str, default: str = "mixed") -> str:
    values: list[Any] = []
    for point in points:
        for manifest in point.manifests:
            if key in manifest:
                values.append(manifest[key])

    if not values:
        return default

    first = values[0]
    if all(value == first for value in values):
        return _normalise_manifest_value(first).label
    return default


def _build_axis(points: list[ParameterSweepRun]) -> tuple[list[float], list[str], bool]:
    if all(point.normalised_value.axis_value is not None for point in points):
        x = [float(point.normalised_value.axis_value) for point in points]
        labels = [point.normalised_value.label for point in points]
        return x, labels, True

    x = [float(i) for i in range(len(points))]
    labels = [point.normalised_value.label for point in points]
    return x, labels, False


def _metric_series(point: ParameterSweepRun, metric_name: str) -> tuple[float, float | None, float | None]:
    spec = METRICS[metric_name]
    value = float(getattr(point, spec.attribute))

    if not spec.use_errorbars:
        return value, None, None

    low = float(point.mean_logical_error_rate_per_round_ci_low)
    high = float(point.mean_logical_error_rate_per_round_ci_high)
    return value, max(0.0, value - low), max(0.0, high - value)


def _plot_grouped_series(
    grouped_points: dict[str, list[ParameterSweepRun]],
    metric_name: str,
    *,
    xlabel: str,
    title: str,
    legend_title: str | None,
    output: Path,
) -> None:
    spec = METRICS[metric_name]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    for series_label, points in sorted(grouped_points.items(), key=lambda item: item[0]):
        points = sorted(points, key=lambda p: p.normalised_value.sort_key)
        x, x_labels, use_numeric_axis = _build_axis(points)

        y_values: list[float] = []
        yerr_low: list[float] = []
        yerr_high: list[float] = []

        for point in points:
            value, low, high = _metric_series(point, metric_name)
            y_values.append(value)
            if low is not None and high is not None:
                yerr_low.append(low)
                yerr_high.append(high)

        if spec.use_errorbars:
            ax.errorbar(
                x,
                y_values,
                yerr=[yerr_low, yerr_high],
                marker="o",
                capsize=4,
                label=series_label,
            )
        else:
            ax.plot(x, y_values, marker="o", label=series_label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(spec.ylabel)
    ax.set_title(title)
    if use_numeric_axis:
        ax.set_xticks(x, x_labels)
    else:
        ax.set_xticks(x, x_labels, rotation=45, ha="right")
    ax.set_yscale(spec.yscale, **(spec.yscale_kwargs or {}))
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(title=legend_title)

    fig.tight_layout()
    save_figure(fig, output)
    plt.close(fig)


def main() -> int:
    setup_matplotlib()

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
        "--secondary-parameter",
        type=str,
        default=None,
        help="Optional manifest key to split the plot into separate curves.",
    )
    parser.add_argument(
        "--metric",
        action="append",
        choices=list(METRICS),
        default=None,
        help="Metric to plot. Repeat the flag to select several. Defaults to all metrics.",
    )
    parser.add_argument(
        "--basis",
        type=str,
        default=None,
        help="Optional basis filter.",
    )
    parser.add_argument(
        "--decode-mode",
        type=str,
        default=None,
        help="Optional decode-mode filter.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Optional worker-count filter.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Optional thread-count filter.",
    )
    parser.add_argument(
        "--n-shots",
        type=int,
        default=None,
        help="Optional shot-count filter.",
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
        "--sparsify-errors",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional sparsification filter.",
    )
    parser.add_argument(
        "--sparsify-base-degree",
        type=int,
        default=None,
        help="Optional sparsify base-degree filter.",
    )
    parser.add_argument(
        "--sparsify-max-degree",
        type=int,
        default=None,
        help="Optional sparsify max-degree filter.",
    )
    parser.add_argument(
        "--sparsify-reactivate-limit",
        type=int,
        default=None,
        help="Optional sparsify reactivate-limit filter.",
    )
    args = parser.parse_args()

    expanded_run_dirs = expand_run_dirs(args.run_dirs)
    args.run_dirs = expanded_run_dirs

    grouped, run_records, resolved_parameter_name = _group_runs_by_parameter(args.run_dirs, args)
    if not grouped:
        print("No matching runs found.")
        return 1

    varying_fields = {resolved_parameter_name}
    if args.secondary_parameter is not None:
        secondary_parameter_name = _resolve_parameter_name(
            args.secondary_parameter,
            run_records[0][1],
        )
        varying_fields.add(secondary_parameter_name)

    check_manifests_consistent(
        [manifest for _, manifest in run_records],
        varying_fields=varying_fields,
    )

    series_points: dict[str, list[ParameterSweepRun]] = {}

    for runs in grouped.values():
        primary_point = _summarise_group(
            runs[0][3],
            [(r, m, rows, p, s) for (r, m, rows, p, s) in runs],
        )
        secondary_point = runs[0][4]
        series_label = secondary_point.label if secondary_point is not None else "all"
        series_points.setdefault(series_label, []).append(primary_point)

    for pts in series_points.values():
        pts.sort(key=lambda p: p.normalised_value.sort_key)

    all_points = [point for pts in series_points.values() for point in pts]
    all_points.sort(key=lambda p: p.normalised_value.sort_key)

    print(
        "parameter_value,label,mean_decode_time_seconds,mean_shots_per_second,mean_logical_error_rate_per_round,run_count"
    )
    for point in all_points:
        parameter_value = (
            format_number(point.normalised_value.axis_value)
            if point.normalised_value.axis_value is not None
            else point.normalised_value.label
        )
        print(
            f"{parameter_value},"
            f"{point.normalised_value.label},"
            f"{point.mean_decode_time_seconds:.6f},"
            f"{point.mean_shots_per_second:.2f},"
            f"{point.mean_logical_error_rate_per_round:.8e},"
            f"{len(point.run_dirs)}"
        )

    shots_label = _shared_manifest_value(all_points, "n_shots")
    basis_label = _shared_manifest_value(all_points, "basis")
    parameter_label = safe_filename_component(args.parameter)
    shots_label_safe = safe_filename_component(shots_label)
    basis_label_safe = safe_filename_component(basis_label)

    selected_metrics = list(dict.fromkeys(args.metric if args.metric is not None else list(METRICS)))
    saved_paths: list[Path] = []

    for metric_name in selected_metrics:
        output_path = args.output_dir / f"{parameter_label}_{shots_label_safe}_{basis_label_safe}_{metric_name}.png"
        _plot_grouped_series(
            series_points,
            metric_name,
            xlabel=args.parameter,
            title=f"{args.parameter} sweep: {METRICS[metric_name].title_fragment}",
            legend_title=args.secondary_parameter,
            output=output_path,
        )
        saved_paths.append(output_path)

    print()
    for path in saved_paths:
        print(f"Saved: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
