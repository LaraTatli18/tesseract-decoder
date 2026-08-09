from __future__ import annotations

"""Plot logical error rate versus physical error rate for surface-code runs.

This script reads a single benchmark run directory, loads its `results.csv` and
`manifest.json`, and plots logical error rate per round against physical error
rate with one curve per code distance.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plot_utils import load_manifest, save_figure, setup_matplotlib


def main() -> int:
    setup_matplotlib()

    parser = argparse.ArgumentParser(
        description="Plot logical error rate versus physical error rate for surface-code runs.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to run directory containing results.csv and manifest.json",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    df = pd.read_csv(run_dir / "results.csv")

    manifest = load_manifest(run_dir / "manifest.json")

    fig, ax = plt.subplots(figsize=(7, 5))

    for distance in sorted(df["distance"].unique()):
        subset = df[df["distance"] == distance].sort_values("physical_error_rate")

        ax.plot(
            subset["physical_error_rate"],
            subset["logical_error_rate_per_round"],
            marker="o",
            label=f"d={distance}",
        )

        # ax.errorbar(
        #     subset["physical_error_rate"],
        #     subset["logical_error_rate_per_round"],
        #     yerr=[
        #         subset["logical_error_rate_per_round"] - subset["logical_error_rate_ci_low"],
        #         subset["logical_error_rate_ci_high"] - subset["logical_error_rate_per_round"],
        #     ],
        #     marker="o",
        # )

    ax.set_title(f"{manifest['basis']}\n{manifest['n_shots']:,} shots")
    ax.set_xlabel("Physical error rate")
    ax.set_ylabel("Logical error rate per round")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()

    out = run_dir / "logical_error_rate.png"
    save_figure(fig, out)
    plt.close(fig)

    print(f"Saved plot to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
