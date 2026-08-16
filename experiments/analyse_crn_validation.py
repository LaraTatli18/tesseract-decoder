from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CRNRun:
    """Files and metadata belonging to one CRN validation run."""

    seed: int
    det_beam: int
    pqlimit: int
    beam_climbing: bool
    run_dir: Path
    detections_file: Path
    observables_file: Path
    shot_correct_file: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse paired common-random-number decoder validation runs."
    )

    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        required=True,
        help=(
            "Directory containing CRN validation runs. Can be supplied "
            "multiple times."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/crn_validation_pairwise.csv"),
        help="Output CSV for pairwise CRN statistics.",
    )

    parser.add_argument(
        "--weight-bin-width",
        type=int,
        default=10,
        help="Width of syndrome-weight bins used for failure-rate analysis.",
    )

    return parser.parse_args()


def find_single_file(directory: Path, pattern: str) -> Path:
    """Find exactly one file matching a pattern."""

    matches = list(directory.glob(pattern))

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one file matching {pattern!r} in "
            f"{directory}, found {len(matches)}."
        )

    return matches[0]


def load_runs(root: Path) -> list[CRNRun]:
    """Discover CRN runs using their manifest files."""

    runs: list[CRNRun] = []

    for manifest_file in sorted(root.rglob("manifest.json")):
        run_dir = manifest_file.parent

        with manifest_file.open() as f:
            manifest = json.load(f)

        if not manifest.get("crn", False):
            continue

        samples_dir = run_dir / "samples"

        if not samples_dir.exists():
            print(f"Skipping {run_dir}: no samples directory.")
            continue

        detections_file = find_single_file(
            samples_dir,
            "*_detections.npy",
        )

        observables_file = find_single_file(
            samples_dir,
            "*_observables.npy",
        )

        shot_correct_file = find_single_file(
            samples_dir,
            "*_shot_correct.npy",
        )

        runs.append(
            CRNRun(
                seed=int(manifest["crn_seed"]),
                det_beam=int(manifest["det_beam"]),
                pqlimit=int(manifest["pqlimit"]),
                beam_climbing=bool(manifest["beam_climbing"]),
                run_dir=run_dir,
                detections_file=detections_file,
                observables_file=observables_file,
                shot_correct_file=shot_correct_file,
            )
        )

    return runs


def print_run_summary(runs: list[CRNRun]) -> None:
    """Print all discovered CRN runs."""

    print("\nDiscovered CRN runs")
    print("===================")

    for run in sorted(
        runs,
        key=lambda r: (
            r.seed,
            r.det_beam,
            r.pqlimit,
            r.beam_climbing,
        ),
    ):
        print(
            f"seed={run.seed}  "
            f"beam={run.det_beam:2d}  "
            f"pq={run.pqlimit:6d}  "
            f"bc={int(run.beam_climbing)}  "
            f"{run.run_dir.name}"
        )

    print(f"\nTotal runs: {len(runs)}")


def group_by_seed(
    runs: list[CRNRun],
) -> dict[int, list[CRNRun]]:
    """Group runs by CRN seed."""

    grouped: dict[int, list[CRNRun]] = {}

    for run in runs:
        grouped.setdefault(run.seed, []).append(run)

    return grouped


def validate_crn_group(
    seed: int,
    runs: list[CRNRun],
) -> None:
    """
    Check that all configurations within a seed saw exactly the same
    detector and observable samples.
    """

    if len(runs) < 2:
        raise RuntimeError(
            f"Seed {seed} contains only {len(runs)} run(s); "
            "cannot perform paired CRN comparison."
        )

    reference = runs[0]

    reference_detections = np.load(
        reference.detections_file
    )
    reference_observables = np.load(
        reference.observables_file
    )

    for run in runs[1:]:
        detections = np.load(
            run.detections_file
        )
        observables = np.load(
            run.observables_file
        )

        if not np.array_equal(
            reference_detections,
            detections,
        ):
            raise RuntimeError(
                f"CRN validation failed for seed {seed}: "
                f"detections differ between "
                f"beam={reference.det_beam}, "
                f"pq={reference.pqlimit}, "
                f"bc={int(reference.beam_climbing)} and "
                f"beam={run.det_beam}, "
                f"pq={run.pqlimit}, "
                f"bc={int(run.beam_climbing)}."
            )

        if not np.array_equal(
            reference_observables,
            observables,
        ):
            raise RuntimeError(
                f"CRN validation failed for seed {seed}: "
                f"observables differ between "
                f"beam={reference.det_beam}, "
                f"pq={reference.pqlimit}, "
                f"bc={int(reference.beam_climbing)} and "
                f"beam={run.det_beam}, "
                f"pq={run.pqlimit}, "
                f"bc={int(run.beam_climbing)}."
            )

    print(
        f"Seed {seed}: CRN samples identical across "
        f"{len(runs)} configurations."
    )


def syndrome_weights(
    detections: np.ndarray,
) -> np.ndarray:
    """Return the number of active detectors in each shot."""

    if detections.ndim != 2:
        raise RuntimeError(
            f"Expected 2D detection array, "
            f"got shape {detections.shape}."
        )

    return detections.sum(axis=1)


def describe_weights(
    weights: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """Calculate syndrome-weight statistics for a subset of shots."""

    selected = weights[mask]

    if len(selected) == 0:
        return {
            "mean_syndrome_weight": float("nan"),
            "median_syndrome_weight": float("nan"),
            "min_syndrome_weight": float("nan"),
            "max_syndrome_weight": float("nan"),
        }

    return {
        "mean_syndrome_weight": float(
            np.mean(selected)
        ),
        "median_syndrome_weight": float(
            np.median(selected)
        ),
        "min_syndrome_weight": float(
            np.min(selected)
        ),
        "max_syndrome_weight": float(
            np.max(selected)
        ),
    }


def analyse_failure_rate_by_weight(
    seed: int,
    run: CRNRun,
    bin_width: int,
) -> list[dict[str, object]]:
    """
    Calculate decoder failure rate as a function of syndrome weight.
    """

    detections = np.load(
        run.detections_file
    )
    correct = np.load(
        run.shot_correct_file
    ).astype(bool)

    if detections.shape[0] != correct.shape[0]:
        raise RuntimeError(
            f"Seed {seed}, beam {run.det_beam}, "
            f"pq {run.pqlimit}: number of detection "
            "samples does not match correctness samples."
        )

    weights = syndrome_weights(
        detections
    ).astype(int)

    if bin_width <= 0:
        raise ValueError(
            "bin_width must be positive."
        )

    max_weight = (
        int(weights.max())
        if len(weights)
        else 0
    )

    rows: list[dict[str, object]] = []

    for lower in range(
        0,
        max_weight + bin_width,
        bin_width,
    ):
        upper = lower + bin_width

        mask = (
            (weights >= lower)
            & (weights < upper)
        )

        count = int(mask.sum())

        if count == 0:
            continue

        failures = int(
            (mask & ~correct).sum()
        )

        rows.append(
            {
                "analysis": "failure_rate_by_weight",
                "seed": seed,
                "beam": run.det_beam,
                "pqlimit": run.pqlimit,
                "beam_climbing": run.beam_climbing,
                "weight_lower": lower,
                "weight_upper": upper,
                "shots": count,
                "failures": failures,
                "failure_rate": failures / count,
            }
        )

    return rows


def analyse_rescue_probability_by_weight(
    seed: int,
    run_a: CRNRun,
    run_b: CRNRun,
    bin_width: int,
) -> list[dict[str, object]]:
    """
    Calculate how often B rescues A's failures as a function of
    syndrome weight.
    """

    detections_a = np.load(
        run_a.detections_file
    )
    detections_b = np.load(
        run_b.detections_file
    )

    correct_a = np.load(
        run_a.shot_correct_file
    ).astype(bool)

    correct_b = np.load(
        run_b.shot_correct_file
    ).astype(bool)

    if not np.array_equal(
        detections_a,
        detections_b,
    ):
        raise RuntimeError(
            f"Seed {seed}: CRN detection arrays "
            "differ between paired runs."
        )

    if correct_a.shape != correct_b.shape:
        raise RuntimeError(
            f"Correctness arrays have different shapes: "
            f"{correct_a.shape} vs {correct_b.shape}."
        )

    weights = syndrome_weights(
        detections_a
    ).astype(int)

    a_failed = ~correct_a
    b_rescued = a_failed & correct_b

    if bin_width <= 0:
        raise ValueError(
            "bin_width must be positive."
        )

    max_weight = (
        int(weights.max())
        if len(weights)
        else 0
    )

    rows: list[dict[str, object]] = []

    for lower in range(
        0,
        max_weight + bin_width,
        bin_width,
    ):
        upper = lower + bin_width

        weight_mask = (
            (weights >= lower)
            & (weights < upper)
        )

        denominator = int(
            (weight_mask & a_failed).sum()
        )

        rescued = int(
            (weight_mask & b_rescued).sum()
        )

        if denominator == 0:
            continue

        rows.append(
            {
                "analysis": "rescue_probability_by_weight",
                "seed": seed,
                "beam_a": run_a.det_beam,
                "pqlimit_a": run_a.pqlimit,
                "beam_climbing_a": run_a.beam_climbing,
                "beam_b": run_b.det_beam,
                "pqlimit_b": run_b.pqlimit,
                "beam_climbing_b": run_b.beam_climbing,
                "weight_lower": lower,
                "weight_upper": upper,
                "a_failures": denominator,
                "b_rescued": rescued,
                "rescue_probability": rescued / denominator,
            }
        )

    return rows


def find_representative_shots(
    seed: int,
    run_a: CRNRun,
    run_b: CRNRun,
    top_n: int = 10,
) -> list[dict[str, object]]:
    """
    Find high-syndrome-weight shots where B rescues A.
    """

    detections_a = np.load(
        run_a.detections_file
    )
    detections_b = np.load(
        run_b.detections_file
    )

    correct_a = np.load(
        run_a.shot_correct_file
    ).astype(bool)

    correct_b = np.load(
        run_b.shot_correct_file
    ).astype(bool)

    if not np.array_equal(
        detections_a,
        detections_b,
    ):
        raise RuntimeError(
            f"Seed {seed}: CRN detection arrays "
            "differ between paired runs."
        )

    weights = syndrome_weights(
        detections_a
    ).astype(int)

    rescued = (
        (~correct_a)
        & correct_b
    )

    indices = np.flatnonzero(
        rescued
    )

    indices = indices[
        np.argsort(
            weights[indices]
        )[::-1]
    ]

    indices = indices[:top_n]

    rows: list[dict[str, object]] = []

    for shot_index in indices:
        rows.append(
            {
                "seed": seed,
                "beam_a": run_a.det_beam,
                "pqlimit_a": run_a.pqlimit,
                "beam_climbing_a": run_a.beam_climbing,
                "beam_b": run_b.det_beam,
                "pqlimit_b": run_b.pqlimit,
                "beam_climbing_b": run_b.beam_climbing,
                "shot_index": int(shot_index),
                "syndrome_weight": int(
                    weights[shot_index]
                ),
                "a_correct": False,
                "b_correct": True,
            }
        )

    return rows


def analyse_pair(
    seed: int,
    run_a: CRNRun,
    run_b: CRNRun,
) -> list[dict[str, object]]:
    """
    Perform shot-by-shot paired analysis of two decoder configurations.
    """

    detections_a = np.load(
        run_a.detections_file
    )
    detections_b = np.load(
        run_b.detections_file
    )

    if not np.array_equal(
        detections_a,
        detections_b,
    ):
        raise RuntimeError(
            f"Cannot compare seed {seed}: "
            "detection arrays differ."
        )

    correct_a = np.load(
        run_a.shot_correct_file
    ).astype(bool)

    correct_b = np.load(
        run_b.shot_correct_file
    ).astype(bool)

    if correct_a.shape != correct_b.shape:
        raise RuntimeError(
            f"Correctness arrays have different shapes: "
            f"{correct_a.shape} vs {correct_b.shape}."
        )

    if (
        detections_a.shape[0]
        != correct_a.shape[0]
    ):
        raise RuntimeError(
            "Number of detection samples does not match "
            "number of correctness samples."
        )

    weights = syndrome_weights(
        detections_a
    )

    categories = {
        "both_succeed": (
            correct_a
            & correct_b
        ),
        "a_succeeds_b_fails": (
            correct_a
            & ~correct_b
        ),
        "a_fails_b_succeeds": (
            ~correct_a
            & correct_b
        ),
        "both_fail": (
            ~correct_a
            & ~correct_b
        ),
    }

    rows: list[dict[str, object]] = []

    print()
    print(
        f"Seed {seed}: "
        f"beam {run_a.det_beam}/"
        f"pq {run_a.pqlimit}/"
        f"bc{int(run_a.beam_climbing)} "
        f"vs "
        f"beam {run_b.det_beam}/"
        f"pq {run_b.pqlimit}/"
        f"bc{int(run_b.beam_climbing)}"
    )

    print("-" * 80)

    for category, mask in categories.items():
        count = int(
            mask.sum()
        )

        stats = describe_weights(
            weights,
            mask,
        )

        print(
            f"{category:22s}: "
            f"{count:6d}  "
            f"mean syndrome weight="
            f"{stats['mean_syndrome_weight']:.3f}"
        )

        rows.append(
            {
                "seed": seed,
                "beam_a": run_a.det_beam,
                "pqlimit_a": run_a.pqlimit,
                "beam_climbing_a": run_a.beam_climbing,
                "beam_b": run_b.det_beam,
                "pqlimit_b": run_b.pqlimit,
                "beam_climbing_b": run_b.beam_climbing,
                "category": category,
                "count": count,
                **stats,
            }
        )

    failures_a = int(
        (~correct_a).sum()
    )

    failures_b = int(
        (~correct_b).sum()
    )

    rescued_by_b = int(
        (
            ~correct_a
            & correct_b
        ).sum()
    )

    regressions_b = int(
        (
            correct_a
            & ~correct_b
        ).sum()
    )

    print()
    print(
        f"A failures             : "
        f"{failures_a}"
    )

    print(
        f"B failures             : "
        f"{failures_b}"
    )

    print(
        f"Shots rescued by B     : "
        f"{rescued_by_b}"
    )

    print(
        f"Shots regressed under B: "
        f"{regressions_b}"
    )

    print(
        f"Net failure improvement: "
        f"{failures_a - failures_b}"
    )

    return rows


def write_results(
    rows: list[dict[str, object]],
    output: Path,
) -> None:
    """Write an analysis table to CSV."""

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        print(
            f"No rows to write for: "
            f"{output}"
        )
        return

    fieldnames = list(
        rows[0].keys()
    )

    with output.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    print(
        f"Saved analysis to: "
        f"{output}"
    )


def main() -> None:
    args = parse_args()

    runs: list[CRNRun] = []

    for run_dir in args.run_dir:
        runs.extend(
            load_runs(
                run_dir
            )
        )

    if not runs:
        raise RuntimeError(
            f"No CRN runs found under "
            f"{args.run_dir}."
        )

    print_run_summary(
        runs
    )

    grouped = group_by_seed(
        runs
    )

    print(
        "\nValidating CRN samples"
    )
    print(
        "======================"
    )

    for seed, seed_runs in sorted(
        grouped.items()
    ):
        validate_crn_group(
            seed,
            seed_runs,
        )

    all_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    rescue_rows: list[dict[str, object]] = []
    representative_rows: list[dict[str, object]] = []

    print(
        "\nPaired decoder analysis"
    )
    print(
        "======================="
    )

    for seed, seed_runs in sorted(
        grouped.items()
    ):
        configurations = {
            (
                run.det_beam,
                run.pqlimit,
                run.beam_climbing,
            ): run
            for run in seed_runs
        }

        # Failure probability as a function of syndrome
        # weight for every individual configuration.
        for run in seed_runs:
            weight_rows.extend(
                analyse_failure_rate_by_weight(
                    seed=seed,
                    run=run,
                    bin_width=args.weight_bin_width,
                )
            )

        desired_pairs = [
            # Documented default -> beam-climbing ablation.
            (
                (5, 200000, False),
                (5, 200000, True),
            ),

            # Documented default -> tuned configurations.
            (
                (5, 200000, False),
                (10, 75000, True),
            ),
            (
                (5, 200000, False),
                (15, 20000, True),
            ),
            (
                (5, 200000, False),
                (25, 50000, True),
            ),

            # Beam/pqlimit tuning with beam climbing enabled.
            (
                (10, 75000, True),
                (15, 20000, True),
            ),
            (
                (15, 20000, True),
                (25, 50000, True),
            ),
            (
                (10, 75000, True),
                (25, 50000, True),
            ),
        ]

        for config_a, config_b in desired_pairs:
            if config_a not in configurations:
                print(
                    f"Skipping seed {seed}: "
                    f"missing configuration "
                    f"{config_a}."
                )
                continue

            if config_b not in configurations:
                print(
                    f"Skipping seed {seed}: "
                    f"missing configuration "
                    f"{config_b}."
                )
                continue

            run_a = configurations[
                config_a
            ]
            run_b = configurations[
                config_b
            ]

            pair_rows = analyse_pair(
                seed=seed,
                run_a=run_a,
                run_b=run_b,
            )

            all_rows.extend(
                pair_rows
            )

            rescue_rows.extend(
                analyse_rescue_probability_by_weight(
                    seed=seed,
                    run_a=run_a,
                    run_b=run_b,
                    bin_width=args.weight_bin_width,
                )
            )

            representative_rows.extend(
                find_representative_shots(
                    seed=seed,
                    run_a=run_a,
                    run_b=run_b,
                    top_n=10,
                )
            )

    write_results(
        all_rows,
        args.output,
    )

    weight_output = args.output.with_name(
        "crn_validation_failure_by_weight.csv"
    )

    rescue_output = args.output.with_name(
        "crn_validation_rescue_by_weight.csv"
    )

    representative_output = args.output.with_name(
        "crn_validation_representative_shots.csv"
    )

    write_results(
        weight_rows,
        weight_output,
    )

    write_results(
        rescue_rows,
        rescue_output,
    )

    write_results(
        representative_rows,
        representative_output,
    )


if __name__ == "__main__":
    main()
