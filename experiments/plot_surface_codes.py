from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_csv(ROOT / "experiments" / "results" / "surface_code_x_10k.csv")
df = df[df["physical_error_rate"].isin([0.0005, 0.001, 0.002])]

print(df.columns.tolist())
print(df)

print(
    df.groupby(
        ["distance", "physical_error_rate"]
    ).size()
)

plt.figure(figsize=(7, 5))

for distance in sorted(df["distance"].unique()):
    subset = df[df["distance"] == distance].sort_values("physical_error_rate")

    plt.plot(
        subset["physical_error_rate"],
        subset["logical_error_rate_per_round"],
        marker="o",
        label=f"d={distance}",
    )

plt.xlabel("Physical error rate")
plt.ylabel("Logical error rate per round")
plt.yscale("log")
plt.grid(True, which="both", linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()

out = ROOT / "experiments" / "logical_error_rate_10kshots.png"
plt.savefig(out, dpi=300)
print(f"Saved plot to {out}")





