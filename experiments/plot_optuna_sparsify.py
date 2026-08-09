from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path.home() / "PycharmProjects" / "tesseract-decoder"
STUDY = (
    ROOT
    / "experiments"
    / "runs"
    / "surface_code_X_sparsify_optuna_d11_p0p002_beam20_100k_v5"
)
OUT = ROOT / "experiments" / "plots" / "optuna_sparsify_v5"
OUT.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")


def load_rows() -> list[dict[str, str]]:
    path = STUDY / "results.csv"
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value in ("", "None", None):
        return float("nan")
    return float(value)


def as_int(value: str) -> int:
    return int(float(value))


def as_int_label(value: str) -> str:
    try:
        return str(int(float(value)))
    except ValueError:
        return value


def fmt_num(value: float) -> str:
    if np.isnan(value):
        return "nan"
    if value >= 100 or value == int(value):
        return f"{value:.0f}"
    if value >= 1:
        return f"{value:.2f}"
    return f"{value:.2e}"


def pareto_front(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    front: list[dict[str, str]] = []

    def dominated(a: dict[str, str], b: dict[str, str]) -> bool:
        br = as_float(b, "objective_runtime")
        ar = as_float(a, "objective_runtime")
        bq = as_float(b, "objective_quality")
        aq = as_float(a, "objective_quality")
        return (br <= ar and bq <= aq) and (br < ar or bq < aq)

    for a in rows:
        if not any(dominated(a, b) for b in rows if b is not a):
            front.append(a)
    return front


def count_values(rows: list[dict[str, str]], key: str) -> Counter[str]:
    c: Counter[str] = Counter()
    for row in rows:
        c[row[key]] += 1
    return c


def grouped_numeric(rows: list[dict[str, str]], key: str, metric: str) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(as_float(row, metric))
    return dict(grouped)


def print_usage_summary(rows: list[dict[str, str]], front: list[dict[str, str]]) -> None:
    print(f"Loaded {len(rows)} trials")
    print(f"Pareto front size: {len(front)}")
    print()

    for key, title in [
        ("sparsify_base_degree", "base degree"),
        ("sparsify_max_degree", "max degree"),
        ("sparsify_reactivate_limit", "reactivate limit"),
    ]:
        all_counts = count_values(rows, key)
        pareto_counts = count_values(front, key)
        print(f"{title} usage")
        for label in sorted(
            set(all_counts) | set(pareto_counts),
            key=lambda x: (float(x), x) if x != "-1" else (-1e9, x),
        ):
            print(
                f"  {label:>4}  all={all_counts.get(label, 0):>3}  pareto={pareto_counts.get(label, 0):>3}"
            )
        print()


def plot_runtime_vs_quality(rows: list[dict[str, str]], front: list[dict[str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6))

    base_colors = {2: "tab:blue", 3: "tab:orange", 4: "tab:green"}
    react_sizes = {-1: 45, 16: 60, 32: 80, 64: 100, 128: 120, 256: 140}

    for row in rows:
        base = as_int(row["sparsify_base_degree"])
        react = as_int(row["sparsify_reactivate_limit"])
        ax.scatter(
            as_float(row, "objective_runtime"),
            as_float(row, "objective_quality"),
            s=react_sizes.get(react, 55),
            c=base_colors.get(base, "gray"),
            alpha=0.35,
            edgecolors="none",
        )

    front_sorted = sorted(front, key=lambda r: as_float(r, "objective_runtime"))
    ax.plot(
        [as_float(r, "objective_runtime") for r in front_sorted],
        [as_float(r, "objective_quality") for r in front_sorted],
        color="black",
        linewidth=1.2,
        alpha=0.75,
    )
    ax.scatter(
        [as_float(r, "objective_runtime") for r in front_sorted],
        [as_float(r, "objective_quality") for r in front_sorted],
        s=100,
        facecolors="none",
        edgecolors="black",
        linewidths=1.6,
        label="Pareto front",
        zorder=3,
    )

    for row in front_sorted:
        ax.annotate(
            f"t{row['trial_number']}",
            (as_float(row, "objective_runtime"), as_float(row, "objective_quality")),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Runtime (s)")
    ax.set_ylabel("Logical error rate per round")
    ax.set_title("Optuna sparsification study: runtime vs quality")

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label="base=2",
                   markerfacecolor="tab:blue", markersize=8),
        plt.Line2D([0], [0], marker="o", color="w", label="base=3",
                   markerfacecolor="tab:orange", markersize=8),
        plt.Line2D([0], [0], marker="o", color="w", label="base=4",
                   markerfacecolor="tab:green", markersize=8),
        plt.Line2D([0], [0], marker="o", color="w", label="Pareto front",
                   markerfacecolor="none", markeredgecolor="black", markersize=8),
    ]
    ax.legend(handles=handles, loc="best", frameon=True)

    fig.tight_layout()
    fig.savefig(OUT / "runtime_vs_quality.png", dpi=200)
    fig.savefig(OUT / "runtime_vs_quality.pdf")
    plt.close(fig)


def plot_parameter_counts(rows: list[dict[str, str]], front: list[dict[str, str]]) -> None:
    def count(key: str, data: list[dict[str, str]]) -> Counter[str]:
        c: Counter[str] = Counter()
        for row in data:
            c[row[key]] += 1
        return c

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)

    for ax, key, title in [
        (axes[0], "sparsify_base_degree", "Base degree"),
        (axes[1], "sparsify_max_degree", "Max degree"),
        (axes[2], "sparsify_reactivate_limit", "Reactivate limit"),
    ]:
        full = count(key, rows)
        pareto = count(key, front)

        labels = sorted(
            set(full) | set(pareto),
            key=lambda x: (float(x), x) if x != "-1" else (-1e9, x),
        )
        x = range(len(labels))
        full_vals = [full.get(label, 0) for label in labels]
        pareto_vals = [pareto.get(label, 0) for label in labels]

        ax.bar(x, full_vals, color="lightgray", label="All trials")
        ax.bar(x, pareto_vals, color="tab:blue", alpha=0.85, label="Pareto")
        ax.set_xticks(list(x))
        ax.set_xticklabels([as_int_label(label) for label in labels], rotation=45, ha="right")
        ax.set_title(title)
        ax.set_ylabel("Count")

    axes[0].legend(frameon=True)
    fig.suptitle("Parameter usage in Optuna study")
    fig.savefig(OUT / "parameter_counts.png", dpi=200)
    fig.savefig(OUT / "parameter_counts.pdf")
    plt.close(fig)


def plot_parameter_effects(rows: list[dict[str, str]], front: list[dict[str, str]]) -> None:
    configs = [
        ("sparsify_base_degree", "Base degree", [2, 3, 4]),
        ("sparsify_max_degree", "Max degree", [-1, 4, 5, 6, 8]),
        ("sparsify_reactivate_limit", "Reactivate limit", [-1, 16, 32, 64, 128, 256]),
    ]

    for metric, metric_label in [
        ("objective_runtime", "Runtime (s)"),
        ("objective_quality", "Logical error rate per round"),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), constrained_layout=True)

        for ax, (key, title, preferred_order) in zip(axes, configs):
            grouped = grouped_numeric(rows, key, metric)
            labels = [v for v in preferred_order if str(v) in grouped]
            data = [grouped[str(v)] for v in labels]

            ax.boxplot(data, showfliers=False)
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels([as_int_label(str(v)) for v in labels], rotation=45, ha="right")
            ax.set_title(title)
            ax.set_ylabel(metric_label)

            if metric == "objective_runtime":
                ax.set_yscale("log")

            pareto_grouped = grouped_numeric(front, key, metric)
            for i, label in enumerate(labels, start=1):
                vals = pareto_grouped.get(str(label), [])
                if vals:
                    ax.scatter([i] * len(vals), vals, color="tab:red", s=24, zorder=3)

        fig.suptitle(f"{metric_label} by parameter value")
        filename = "runtime_by_parameter" if metric == "objective_runtime" else "quality_by_parameter"
        fig.savefig(OUT / f"{filename}.png", dpi=200)
        fig.savefig(OUT / f"{filename}.pdf")
        plt.close(fig)


def plot_pareto_front_only(front: list[dict[str, str]]) -> None:
    front_sorted = sorted(front, key=lambda r: as_float(r, "objective_runtime"))
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        [as_float(r, "objective_runtime") for r in front_sorted],
        [as_float(r, "objective_quality") for r in front_sorted],
        marker="o",
        linewidth=2,
        color="tab:blue",
    )
    for row in front_sorted:
        ax.annotate(
            f"t{row['trial_number']}\nB{row['sparsify_base_degree']} M{row['sparsify_max_degree']} R{row['sparsify_reactivate_limit']}",
            (as_float(row, "objective_runtime"), as_float(row, "objective_quality")),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Runtime (s)")
    ax.set_ylabel("Logical error rate per round")
    ax.set_title("Pareto front of sparsification study")
    fig.tight_layout()
    fig.savefig(OUT / "pareto_front.png", dpi=200)
    fig.savefig(OUT / "pareto_front.pdf")
    plt.close(fig)


def plot_heatmaps(rows: list[dict[str, str]]) -> None:
    base_order = sorted({as_int(r["sparsify_base_degree"]) for r in rows})
    max_order = sorted({as_int(r["sparsify_max_degree"]) for r in rows})

    def aggregate(metric: str) -> np.ndarray:
        arr = np.full((len(base_order), len(max_order)), np.nan, dtype=float)
        for i, base in enumerate(base_order):
            for j, max_degree in enumerate(max_order):
                vals = [
                    as_float(r, metric)
                    for r in rows
                    if as_int(r["sparsify_base_degree"]) == base
                    and as_int(r["sparsify_max_degree"]) == max_degree
                ]
                if vals:
                    arr[i, j] = float(np.mean(vals))
        return arr

    runtime = aggregate("objective_runtime")
    quality = aggregate("objective_quality")

    for metric_name, data, cmap, cbar_label in [
        ("runtime_heatmap", runtime, "viridis", "Runtime (s)"),
        ("quality_heatmap", quality, "magma_r", "Logical error rate per round"),
    ]:
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        masked = np.ma.masked_invalid(data)
        im = ax.imshow(masked, origin="lower", aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(max_order)))
        ax.set_xticklabels([as_int_label(str(v)) for v in max_order])
        ax.set_yticks(range(len(base_order)))
        ax.set_yticklabels([str(v) for v in base_order])
        ax.set_xlabel("Max degree")
        ax.set_ylabel("Base degree")
        ax.set_title(metric_name.replace("_", " ").title())

        for i, base in enumerate(base_order):
            for j, max_degree in enumerate(max_order):
                value = data[i, j]
                if not np.isnan(value):
                    ax.text(j, i, fmt_num(value), ha="center", va="center", color="white", fontsize=8)

        fig.colorbar(im, ax=ax, label=cbar_label)
        fig.tight_layout()
        fig.savefig(OUT / f"{metric_name}.png", dpi=200)
        fig.savefig(OUT / f"{metric_name}.pdf")
        plt.close(fig)


def plot_reactivate_sweep(rows: list[dict[str, str]], front: list[dict[str, str]]) -> None:
    base = 2
    subsets = {
        "max4": [r for r in rows if as_int(r["sparsify_base_degree"]) == base and as_int(r["sparsify_max_degree"]) == 4],
        "max5": [r for r in rows if as_int(r["sparsify_base_degree"]) == base and as_int(r["sparsify_max_degree"]) == 5],
        "max6": [r for r in rows if as_int(r["sparsify_base_degree"]) == base and as_int(r["sparsify_max_degree"]) == 6],
        "no_max": [r for r in rows if as_int(r["sparsify_base_degree"]) == base and as_int(r["sparsify_max_degree"]) == -1],
    }

    react_order = [-1, 16, 32, 64, 128, 256]
    colors = {
        "max4": "tab:blue",
        "max5": "tab:orange",
        "max6": "tab:green",
        "no_max": "tab:red",
    }
    labels = {
        "max4": "base=2, max=4",
        "max5": "base=2, max=5",
        "max6": "base=2, max=6",
        "no_max": "base=2, max=-1",
    }

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax2 = ax1.twinx()

    for key, subset in subsets.items():
        if not subset:
            continue
        grouped_runtime = grouped_numeric(subset, "sparsify_reactivate_limit", "objective_runtime")
        grouped_quality = grouped_numeric(subset, "sparsify_reactivate_limit", "objective_quality")
        xs = [v for v in react_order if str(v) in grouped_runtime]
        if not xs:
            continue
        runtimes = [float(np.mean(grouped_runtime[str(v)])) for v in xs]
        qualities = [float(np.mean(grouped_quality[str(v)])) for v in xs]

        ax1.plot(xs, runtimes, marker="o", color=colors[key], label=labels[key])
        ax2.plot(xs, qualities, marker="s", linestyle="--", color=colors[key], alpha=0.7)

    pareto_subset = [r for r in front if as_int(r["sparsify_base_degree"]) == base]
    if pareto_subset:
        ax1.scatter(
            [as_int(r["sparsify_reactivate_limit"]) for r in pareto_subset],
            [as_float(r, "objective_runtime") for r in pareto_subset],
            color="black",
            s=55,
            zorder=4,
            label="Pareto trials (runtime)",
        )
        ax2.scatter(
            [as_int(r["sparsify_reactivate_limit"]) for r in pareto_subset],
            [as_float(r, "objective_quality") for r in pareto_subset],
            color="black",
            s=35,
            zorder=4,
            marker="x",
            label="Pareto trials (quality)",
        )

    ax1.set_xlabel("Reactivate limit")
    ax1.set_ylabel("Runtime (s)")
    ax1.set_yscale("log")
    ax2.set_ylabel("Logical error rate per round")
    ax2.set_yscale("log")
    ax1.set_title("Reactivate-limit sweep for base=2")
    ax1.set_xticks(react_order)
    ax1.set_xticklabels([as_int_label(str(v)) for v in react_order])
    ax1.legend(loc="upper left", frameon=True)
    ax2.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(OUT / "reactivate_sweep_base2.png", dpi=200)
    fig.savefig(OUT / "reactivate_sweep_base2.pdf")
    plt.close(fig)


def main() -> int:
    rows = load_rows()
    front = pareto_front(rows)

    print_usage_summary(rows, front)

    plot_runtime_vs_quality(rows, front)
    plot_parameter_counts(rows, front)
    plot_parameter_effects(rows, front)
    plot_pareto_front_only(front)
    plot_heatmaps(rows)
    plot_reactivate_sweep(rows, front)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
