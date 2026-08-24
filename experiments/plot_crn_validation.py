from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd


CONFIGS = [
    {
        "key": "default",
        "beam": 5,
        "pqlimit": 200000,
        "bc": False,
        "label": "Default",
    },
    {
        "key": "default_bc",
        "beam": 5,
        "pqlimit": 200000,
        "bc": True,
        "label": "Default + beam climbing",
    },
    {
        "key": "beam10",
        "beam": 10,
        "pqlimit": 75000,
        "bc": True,
        "label": "Beam 10 / pqlimit 75k",
    },
    {
        "key": "beam15",
        "beam": 15,
        "pqlimit": 20000,
        "bc": True,
        "label": "Beam 15 / pqlimit 20k",
    },
    {
        "key": "beam25",
        "beam": 25,
        "pqlimit": 50000,
        "bc": True,
        "label": "Beam 25 / pqlimit 50k",
    },
]


PAIRWISE_COMPARISONS = [
    ("default", "default_bc", "Default -> Default + BC"),
    ("default", "beam15", "Default -> Beam 15 / 20k"),
    ("beam15", "beam25", "Beam 15 / 20k -> Beam 25 / 50k"),
]

MIN_WEIGHT_BIN_SHOTS = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create clear summary plots for the Tesseract CRN study."
    )

    parser.add_argument(
        "--crn-run-dir",
        type=Path,
        default=Path("experiments/runs/crn_local_backup"),
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiments/results"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/plots/crn_figures"),
    )

    parser.add_argument(
        "--optuna-dir",
        type=Path,
        default=Path(
            "experiments/runs/surface_code_x_beam_pqlimit_optuna"
        ),
    )

    return parser.parse_args()


def config_key(row: pd.Series) -> str | None:
    beam = int(row["beam"])
    pqlimit = int(row["pqlimit"])
    bc = bool(row["beam_climbing"])

    for config in CONFIGS:
        if (
            beam == config["beam"]
            and pqlimit == config["pqlimit"]
            and bc == config["bc"]
        ):
            return config["key"]

    return None


def pooled_failure_rate_data(
    path: Path,
) -> pd.DataFrame:
    """Pool the three CRN seeds using total failures / total shots per bin."""

    df = pd.read_csv(path)

    df["config_key"] = df.apply(
        config_key,
        axis=1,
    )

    df = df[
        df["config_key"].notna()
    ].copy()

    grouped = (
        df.groupby(
            [
                "config_key",
                "beam",
                "pqlimit",
                "beam_climbing",
                "weight_lower",
                "weight_upper",
            ],
            as_index=False,
        )
        .agg(
            shots=("shots", "sum"),
            failures=("failures", "sum"),
        )
    )

    grouped["failure_rate"] = (
        grouped["failures"] / grouped["shots"]
    )

    grouped["weight_midpoint"] = (
        grouped["weight_lower"]
        + grouped["weight_upper"]
    ) / 2

    return grouped


def pooled_rescue_data(
    path: Path,
) -> pd.DataFrame:
    """Pool rescue statistics across seeds using total rescued / total failures."""

    df = pd.read_csv(path)

    def key_for_side(
        row: pd.Series,
        side: str,
    ) -> str | None:
        beam = int(
            row[f"beam_{side}"]
        )

        pqlimit = int(
            row[f"pqlimit_{side}"]
        )

        bc = bool(
            row[f"beam_climbing_{side}"]
        )

        for config in CONFIGS:
            if (
                beam == config["beam"]
                and pqlimit == config["pqlimit"]
                and bc == config["bc"]
            ):
                return config["key"]

        return None

    df["config_a"] = df.apply(
        lambda row: key_for_side(row, "a"),
        axis=1,
    )

    df["config_b"] = df.apply(
        lambda row: key_for_side(row, "b"),
        axis=1,
    )

    df = df[
        df["config_a"].notna()
        & df["config_b"].notna()
    ].copy()

    grouped = (
        df.groupby(
            [
                "config_a",
                "config_b",
                "weight_lower",
                "weight_upper",
            ],
            as_index=False,
        )
        .agg(
            a_failures=("a_failures", "sum"),
            b_rescued=("b_rescued", "sum"),
        )
    )

    grouped["rescue_probability"] = (
        grouped["b_rescued"]
        / grouped["a_failures"]
    )

    grouped["weight_midpoint"] = (
        grouped["weight_lower"]
        + grouped["weight_upper"]
    ) / 2

    return grouped


def load_crn_summary(
    results_dir: Path,
) -> pd.DataFrame:
    """Create a compact summary for the five validated configurations."""

    rows: list[dict[str, object]] = []

    run_root_candidates = [
        Path(
            "experiments/runs/crn_local_backup/"
            "tesseract_default_crn_validation"
        ),
        Path(
            "experiments/runs/crn_local_backup/"
            "tesseract_default_bc_crn_validation"
        ),
        Path(
            "experiments/runs/crn_local_backup/"
            "optuna_beam_pqlimit_crn_validation"
        ),
    ]

    for root in run_root_candidates:
        for result_file in root.rglob(
            "results.csv"
        ):
            try:
                run = pd.read_csv(
                    result_file
                ).iloc[0]
            except Exception:
                continue

            row = {
                "beam": int(
                    run["det_beam"]
                ),
                "pqlimit": int(
                    run["pqlimit"]
                ),
                "beam_climbing": bool(
                    run["beam_climbing"]
                ),
                "runtime": float(
                    run["decode_time_seconds"]
                ),
                "ler_round": float(
                    run[
                        "logical_error_rate_per_round"
                    ]
                ),
                "shots": int(
                    run["n_shots"]
                ),
                "failures": int(
                    run["logical_failures"]
                ),
            }

            key = config_key(
                pd.Series(
                    {
                        "beam": row["beam"],
                        "pqlimit": row["pqlimit"],
                        "beam_climbing": row[
                            "beam_climbing"
                        ],
                    }
                )
            )

            if key is None:
                continue

            row["config_key"] = key
            rows.append(row)

    summary = pd.DataFrame(rows)

    if summary.empty:
        raise RuntimeError(
            "Could not find CRN run results."
        )

    summary = (
        summary.groupby(
            [
                "config_key",
                "beam",
                "pqlimit",
                "beam_climbing",
            ],
            as_index=False,
        )
        .agg(
            mean_runtime=(
                "runtime",
                "mean",
            ),
            std_runtime=(
                "runtime",
                "std",
            ),
            mean_ler_round=(
                "ler_round",
                "mean",
            ),
            std_ler_round=(
                "ler_round",
                "std",
            ),
            mean_failures=(
                "failures",
                "mean",
            ),
        )
    )

    return summary


def plot_quality_latency(
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    for config in CONFIGS:
        subset = summary[
            summary["config_key"]
            == config["key"]
        ]

        if subset.empty:
            continue

        row = subset.iloc[0]

        ax.errorbar(
            row["mean_runtime"],
            row["mean_ler_round"],
            xerr=row["std_runtime"],
            yerr=row["std_ler_round"],
            fmt="o",
            capsize=4,
            label=config["label"],
        )

        ax.annotate(
            config["label"],
            (
                row["mean_runtime"],
                row["mean_ler_round"],
            ),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xlabel(
        "Decode time (s)"
    )

    ax.set_ylabel(
        "Logical error rate per round"
    )

    ax.set_title(
        "CRN-validated quality-latency trade-off"
    )

    ax.set_yscale("log")
    ax.grid(
        True,
        alpha=0.25,
    )

    fig.tight_layout()

    output = (
        output_dir
        / "quality_latency_tradeoff.png"
    )

    fig.savefig(
        output,
        dpi=300,
    )

    plt.close(fig)

    print(
        f"Saved: {output}"
    )


def plot_failure_rate_by_weight(
    failure_data: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot pooled failure probability against syndrome weight."""

    data = failure_data[
        failure_data["shots"] >= MIN_WEIGHT_BIN_SHOTS
    ].copy()

    data = data[
        (data["weight_midpoint"] >= 60)
        & (data["weight_midpoint"] <= 135)
    ].copy()

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    for config in CONFIGS:
        subset = data[
            data["config_key"] == config["key"]
        ].copy()

        if subset.empty:
            continue

        ax.plot(
            subset["weight_midpoint"],
            subset["failure_rate"],
            marker="o",
            markersize=4,
            linewidth=1.5,
            label=config["label"],
        )

    ax.set_xlabel(
        "Syndrome Hamming weight"
    )

    ax.set_ylabel(
        "Logical failure probability"
    )

    ax.set_title(
        "Logical failure probability vs syndrome Hamming weight"
    )

    ax.set_yscale("log")
    ax.set_xlim(60, 135)

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend(
        fontsize=9
    )

    fig.tight_layout()

    output = (
        output_dir
        / "failure_rate_vs_syndrome_weight.png"
    )

    fig.savefig(
        output,
        dpi=300,
    )

    plt.close(fig)

    print(
        f"Saved: {output}"
    )


def plot_paired_outcomes(
    results_dir: Path,
    output_dir: Path,
) -> None:
    """Plot aggregated CRN rescues versus regressions for key comparisons."""

    pairwise = pd.read_csv(
        results_dir / "crn_validation_pairwise.csv"
    )

    comparisons = [
        (
            "default",
            "default_bc",
            "Default -> Default + BC",
        ),
        (
            "default",
            "beam15",
            "Default -> Beam 15 / 20k",
        ),
        (
            "beam15",
            "beam25",
            "Beam 15 / 20k -> Beam 25 / 50k",
        ),
    ]

    labels = []
    rescues = []
    regressions = []

    def side_key(
        beam: int,
        pqlimit: int,
        bc: bool,
    ) -> str | None:
        for config in CONFIGS:
            if (
                config["beam"] == beam
                and config["pqlimit"] == pqlimit
                and config["bc"] == bc
            ):
                return config["key"]

        return None

    for key_a, key_b, label in comparisons:
        subset = pairwise.copy()

        subset["config_a"] = subset.apply(
            lambda row: side_key(
                int(row["beam_a"]),
                int(row["pqlimit_a"]),
                bool(row["beam_climbing_a"]),
            ),
            axis=1,
        )

        subset["config_b"] = subset.apply(
            lambda row: side_key(
                int(row["beam_b"]),
                int(row["pqlimit_b"]),
                bool(row["beam_climbing_b"]),
            ),
            axis=1,
        )

        subset = subset[
            (subset["config_a"] == key_a)
            & (subset["config_b"] == key_b)
        ]

        rescue_count = int(
            subset.loc[
                subset["category"]
                == "a_fails_b_succeeds",
                "count",
            ].sum()
        )

        regression_count = int(
            subset.loc[
                subset["category"]
                == "a_succeeds_b_fails",
                "count",
            ].sum()
        )

        labels.append(label)
        rescues.append(rescue_count)
        regressions.append(regression_count)

    x = range(len(labels))
    width = 0.38

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.bar(
        [i - width / 2 for i in x],
        rescues,
        width=width,
        label="A fails -> B succeeds",
    )

    ax.bar(
        [i + width / 2 for i in x],
        regressions,
        width=width,
        label="A succeeds -> B fails",
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel(
        "Number of discordant shots"
    )
    ax.set_title(
        "Paired CRN outcomes: rescues versus regressions"
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    output = (
        output_dir
        / "paired_rescues_vs_regressions.png"
    )

    fig.savefig(
        output,
        dpi=300,
    )

    plt.close(fig)

    print(
        f"Saved: {output}"
    )


def load_optuna_trials(
    optuna_dir: Path,
) -> pd.DataFrame:
    """Load valid beam/pqlimit Optuna trials and recompute Pareto membership."""

    results_file = (
        optuna_dir
        / "results.csv"
    )

    df = pd.read_csv(
        results_file
    )

    required = [
        "decode_time_seconds",
        "logical_error_rate_per_round",
        "det_beam",
        "pqlimit",
    ]

    df = df.dropna(
        subset=required
    ).copy()

    df = df[
        (df["decode_time_seconds"] > 0)
        & (
            df[
                "logical_error_rate_per_round"
            ] > 0
        )
        & (df["pqlimit"] > 0)
    ].copy()

    df["det_beam"] = (
        df["det_beam"].astype(int)
    )

    df["pqlimit"] = (
        df["pqlimit"].astype(int)
    )

    points = df[
        [
            "decode_time_seconds",
            "logical_error_rate_per_round",
        ]
    ].to_numpy()

    is_pareto: list[bool] = []

    for i, (
        time_i,
        ler_i,
    ) in enumerate(points):

        dominated = False

        for j, (
            time_j,
            ler_j,
        ) in enumerate(points):

            if i == j:
                continue

            no_worse = (
                time_j <= time_i
                and ler_j <= ler_i
            )

            strictly_better = (
                time_j < time_i
                or ler_j < ler_i
            )

            if (
                no_worse
                and strictly_better
            ):
                dominated = True
                break

        is_pareto.append(
            not dominated
        )

    df["pareto"] = is_pareto

    return df


def plot_optuna_trials_and_pareto(
    optuna_dir: Path,
    output_dir: Path,
) -> None:
    df = load_optuna_trials(
        optuna_dir
    )

    pareto = df[
        df["pareto"]
    ].sort_values(
        "decode_time_seconds"
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    print(
        f"Optuna trial points plotted: {len(df)}"
    )

    print(
        f"Pareto points highlighted: {len(pareto)}"
    )

    ax.scatter(
        df["decode_time_seconds"],
        df["logical_error_rate_per_round"],
        alpha=0.35,
        s=28,
        label="All Optuna trials",
    )

    ax.plot(
        pareto["decode_time_seconds"],
        pareto["logical_error_rate_per_round"],
        linewidth=2,
        label="Pareto frontier",
    )

    ax.scatter(
        pareto["decode_time_seconds"],
        pareto["logical_error_rate_per_round"],
        s=70,
        marker="o",
        label="Pareto-optimal trials",
    )

    for _, row in pareto.iterrows():
        ax.annotate(
            (
                f"trial {int(row['trial_number'])}: "
                f"b{int(row['det_beam'])}, "
                f"pq{int(row['pqlimit']):,}"
            ),
            (
                row["decode_time_seconds"],
                row["logical_error_rate_per_round"],
            ),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel(
        "Decode time (s)"
    )

    ax.set_ylabel(
        "Logical error rate per round"
    )

    ax.set_title(
        "Optuna beam/pqlimit search: all trials and Pareto frontier"
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    output = (
        output_dir
        / "optuna_trials_pareto_frontier.png"
    )

    fig.savefig(
        output,
        dpi=300,
    )

    plt.close(fig)

    print(
        f"Saved: {output}"
    )


def plot_optuna_beam_pq_ler(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot Optuna beam/PQ coordinates coloured by logical error rate."""

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    scatter = ax.scatter(
        df["det_beam"],
        df["pqlimit"],
        c=df[
            "logical_error_rate_per_round"
        ],
        norm=LogNorm(
            vmin=df[
                "logical_error_rate_per_round"
            ].min(),
            vmax=df[
                "logical_error_rate_per_round"
            ].max(),
        ),
        s=55,
        alpha=0.75,
    )

    pareto = df[
        df["pareto"]
    ]

    ax.scatter(
        pareto["det_beam"],
        pareto["pqlimit"],
        facecolors="none",
        edgecolors="black",
        linewidths=1.5,
        s=120,
        label="Pareto-optimal trial",
    )

    for _, row in pareto.iterrows():
        ax.annotate(
            f"{int(row['trial_number'])}",
            (
                row["det_beam"],
                row["pqlimit"],
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel(
        "Detector beam"
    )

    ax.set_ylabel(
        "Priority-queue limit"
    )

    ax.set_title(
        "Optuna search landscape: logical error rate"
    )

    ax.set_yscale("log")

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    colorbar = fig.colorbar(
        scatter,
        ax=ax,
    )

    colorbar.set_label(
        "Logical error rate per round"
    )

    fig.tight_layout()

    output = (
        output_dir
        / "optuna_beam_pq_logical_error_rate.png"
    )

    fig.savefig(
        output,
        dpi=300,
    )

    plt.close(fig)

    print(
        f"Saved: {output}"
    )


def plot_optuna_beam_pq_runtime(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot Optuna beam/PQ coordinates coloured by decode time."""

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    scatter = ax.scatter(
        df["det_beam"],
        df["pqlimit"],
        c=df[
            "decode_time_seconds"
        ],
        norm=LogNorm(
            vmin=df[
                "decode_time_seconds"
            ].min(),
            vmax=df[
                "decode_time_seconds"
            ].max(),
        ),
        s=55,
        alpha=0.75,
    )

    pareto = df[
        df["pareto"]
    ]

    ax.scatter(
        pareto["det_beam"],
        pareto["pqlimit"],
        facecolors="none",
        edgecolors="black",
        linewidths=1.5,
        s=120,
        label="Pareto-optimal trial",
    )

    ax.set_xlabel(
        "Detector beam"
    )

    ax.set_ylabel(
        "Priority-queue limit"
    )

    ax.set_title(
        "Optuna search landscape: decode time"
    )

    ax.set_yscale("log")

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    colorbar = fig.colorbar(
        scatter,
        ax=ax,
    )

    colorbar.set_label(
        "Decode time (s)"
    )

    fig.tight_layout()

    output = (
        output_dir
        / "optuna_beam_pq_decode_time.png"
    )

    fig.savefig(
        output,
        dpi=300,
    )

    plt.close(fig)

    print(
        f"Saved: {output}"
    )


def plot_optuna_pareto_beam_pq(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Show dominated and Pareto-optimal trials directly in beam/PQ space."""

    dominated = df[
        ~df["pareto"]
    ]

    pareto = df[
        df["pareto"]
    ].copy()

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.scatter(
        dominated["det_beam"],
        dominated["pqlimit"],
        alpha=0.3,
        s=35,
        label="Dominated trial",
    )

    ax.scatter(
        pareto["det_beam"],
        pareto["pqlimit"],
        s=100,
        marker="o",
        label="Pareto-optimal trial",
    )

    for _, row in pareto.iterrows():
        ax.annotate(
            (
                f"trial {int(row['trial_number'])}\n"
                f"LER={row['logical_error_rate_per_round']:.2e}\n"
                f"t={row['decode_time_seconds']:.0f}s"
            ),
            (
                row["det_beam"],
                row["pqlimit"],
            ),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=7,
        )

    ax.set_xlabel(
        "Detector beam"
    )

    ax.set_ylabel(
        "Priority-queue limit"
    )

    ax.set_title(
        "Optuna Pareto trials in beam/PQ space"
    )

    ax.set_yscale("log")

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    output = (
        output_dir
        / "optuna_pareto_beam_pq_space.png"
    )

    fig.savefig(
        output,
        dpi=300,
    )

    plt.close(fig)

    print(
        f"Saved: {output}"
    )


def plot_optuna_pq_response_by_beam(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot individual Optuna LER observations against PQ limit for each beam.

    Optuna samples the beam/PQ space sparsely, so individual observations are
    shown rather than connected into a continuous response curve. Pareto-
    optimal observations are highlighted separately.
    """

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    for beam in sorted(
        df["det_beam"].unique()
    ):
        subset = df[
            df["det_beam"] == beam
        ].copy()

        subset = subset.sort_values(
            "pqlimit"
        )

        ax.scatter(
            subset["pqlimit"],
            subset[
                "logical_error_rate_per_round"
            ],
            s=48,
            alpha=0.65,
            label=f"beam {beam}",
        )

        # Mark the best observed trial for this beam without implying that
        # the intervening PQ values were sampled continuously.
        best = subset.loc[
            subset[
                "logical_error_rate_per_round"
            ].idxmin()
        ]

        ax.scatter(
            [best["pqlimit"]],
            [best[
                "logical_error_rate_per_round"
            ]],
            s=120,
            marker="D",
            facecolors="none",
            edgecolors="black",
            linewidths=1.4,
        )

    pareto = df[
        df["pareto"]
    ]

    ax.scatter(
        pareto["pqlimit"],
        pareto[
            "logical_error_rate_per_round"
        ],
        facecolors="none",
        edgecolors="black",
        linewidths=1.6,
        s=120,
        marker="o",
        label="Pareto-optimal trial",
    )

    for _, row in pareto.iterrows():
        ax.annotate(
            (
                f"b={int(row['det_beam'])}, "
                f"Q={int(row['pqlimit']):,}"
            ),
            (
                row["pqlimit"],
                row[
                    "logical_error_rate_per_round"
                ],
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
        )

    ax.set_xlabel(
        "Priority-queue limit"
    )

    ax.set_ylabel(
        "Logical error rate per round"
    )

    ax.set_title(
        "Optuna sampled response to priority-queue limit"
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend(
        fontsize=8,
        ncol=2,
    )

    fig.tight_layout()

    output = (
        output_dir
        / "optuna_pqlimit_response_by_beam.png"
    )

    fig.savefig(
        output,
        dpi=300,
    )

    plt.close(fig)

    print(
        f"Saved: {output}"
    )


def print_optuna_low_pq_summary(
    df: pd.DataFrame,
) -> None:
    """Print low-PQ and Pareto trials for quick inspection."""

    columns = [
        "trial_number",
        "det_beam",
        "pqlimit",
        "decode_time_seconds",
        "logical_error_rate_per_round",
        "pareto",
    ]

    print()
    print(
        "Optuna trials with pqlimit <= 50k:"
    )

    print(
        df[
            df["pqlimit"] <= 50000
        ][columns]
        .sort_values(
            [
                "det_beam",
                "pqlimit",
                "logical_error_rate_per_round",
            ]
        )
        .to_string(
            index=False
        )
    )

    print()
    print(
        "Pareto-optimal Optuna trials:"
    )

    print(
        df[
            df["pareto"]
        ][columns]
        .sort_values(
            "decode_time_seconds"
        )
        .to_string(
            index=False
        )
    )


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = load_crn_summary(
        args.results_dir
    )

    failure_data = (
        pooled_failure_rate_data(
            args.results_dir
            / "crn_validation_failure_by_weight.csv"
        )
    )

    plot_quality_latency(
        summary,
        args.output_dir,
    )

    plot_failure_rate_by_weight(
        failure_data,
        args.output_dir,
    )

    plot_paired_outcomes(
        args.results_dir,
        args.output_dir,
    )

    optuna_trials = load_optuna_trials(
        args.optuna_dir
    )

    plot_optuna_trials_and_pareto(
        args.optuna_dir,
        args.output_dir,
    )

    plot_optuna_beam_pq_ler(
        optuna_trials,
        args.output_dir,
    )

    plot_optuna_beam_pq_runtime(
        optuna_trials,
        args.output_dir,
    )

    plot_optuna_pareto_beam_pq(
        optuna_trials,
        args.output_dir,
    )

    plot_optuna_pq_response_by_beam(
        optuna_trials,
        args.output_dir,
    )

    print_optuna_low_pq_summary(
        optuna_trials
    )


if __name__ == "__main__":
    main()
