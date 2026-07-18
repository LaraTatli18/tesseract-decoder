from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to run directory containing results.csv",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    df = pd.read_csv(run_dir / "results.csv")

    with (run_dir / "manifest.json").open() as f:
        manifest = json.load(f)

    plt.figure(figsize=(7, 5))

    for distance in sorted(df["distance"].unique()):
        subset = df[df["distance"] == distance].sort_values("physical_error_rate")

        plt.plot(
            subset["physical_error_rate"],
            subset["logical_error_rate_per_round"],
            marker="o",
            label=f"d={distance}",
        )

        # plt.errorbar(
        #     subset["physical_error_rate"],
        #     subset["logical_error_rate_per_round"],
        #     yerr=[
        #         subset["logical_error_rate_per_round"] - subset[
        #             "logical_error_rate_ci_low"],
        #         subset["logical_error_rate_ci_high"] - subset[
        #             "logical_error_rate_per_round"],
        #     ],
        #     marker="o",
        # )

    plt.title(f"{manifest['basis']}\n{manifest['n_shots']:,} shots")
    plt.xlabel("Physical error rate")
    plt.ylabel("Logical error rate per round")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    out = run_dir / "logical_error_rate.png"
    plt.savefig(out, dpi=300)
    plt.close()

    print(f"Saved plot to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
