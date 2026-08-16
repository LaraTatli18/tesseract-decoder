from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CRNRun:
    # Files and metadata in one CRN validation run
    seed: int
    det_beam: int
    pqlimit: int
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
        required=True,
        help="Directory containing the CRN validation runs.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/crn_validation_pairwise.csv"),
        help="Output CSV for pairwise CRN statistics.",
    )

    return parser.parse_args()


def find_single_file(directory: Path, pattern: str) -> Path:
    matches = list(directory.glob(pattern))

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one file matching {pattern!r} in "
            f"{directory}, found {len(matches)}."
        )

    return matches[0]


def load_runs(root: Path) -> list[CRNRun]:
    # Finding CRN runs using their manifest files

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
                run_dir=run_dir,
                detections_file=detections_file,
                observables_file=observables_file,
                shot_correct_file=shot_correct_file,
            )
        )

    return runs


def print_run_summary(runs: list[CRNRun]) -> None:
    print("\nDiscovered CRN runs")
    print("===================")

    for run in sorted(
        runs,
        key=lambda r: (r.seed, r.det_beam, r.pqlimit),
    ):
        print(
            f"seed={run.seed}  "
            f"beam={run.det_beam:2d}  "
            f"pq={run.pqlimit:6d}  "
            f"{run.run_dir.name}"
        )

    print(f"\nTotal runs: {len(runs)}")


def group_by_seed(runs: list[CRNRun]) -> dict[int, list[CRNRun]]:
    grouped: dict[int, list[CRNRun]] = {}

    for run in runs:
        grouped.setdefault(run.seed, []).append(run)

    return grouped


def validate_crn_group(seed: int, runs: list[CRNRun]) -> None:
    # Check all configs within a seed saw exactly the same detector and observable samples

    if len(runs) < 2:
        raise RuntimeError(
            f"Seed {seed} contains only {len(runs)} run(s); "
            "cannot perform paired CRN comparison."
        )

    reference = runs[0]

    reference_detections = np.load(reference.detections_file)
    reference_observables = np.load(reference.observables_file)

    for run in runs[1:]:
        detections = np.load(run.detections_file)
        observables = np.load(run.observables_file)

        if not np.array_equal(
            reference_detections,
            detections,
        ):
            raise RuntimeError(
                f"CRN validation failed for seed {seed}: "
                f"detections differ between "
                f"beam={reference.det_beam}, pq={reference.pqlimit} and "
                f"beam={run.det_beam}, pq={run.pqlimit}."
            )

        if not np.array_equal(
            reference_observables,
            observables,
        ):
            raise RuntimeError(
                f"CRN validation failed for seed {seed}: "
                f"observables differ between "
                f"beam={reference.det_beam}, pq={reference.pqlimit} and "
                f"beam={run.det_beam}, pq={run.pqlimit}."
            )

    print(
        f"Seed {seed}: CRN samples identical across "
        f"{len(runs)} configurations."
    )


def syndrome_weights(detections: np.ndarray) -> np.ndarray:
    # Return number of active detectors in each shot

    if detections.ndim != 2:
        raise RuntimeError(
            f"Expected 2D detection array, got shape {detections.shape}."
        )

    return detections.sum(axis=1)


def describe_weights(
    weights: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    # Syndrome-weighted statistics

    selected = weights[mask]

    if len(selected) == 0:
        return {
            "mean_syndrome_weight": float("nan"),
            "median_syndrome_weight": float("nan"),
            "min_syndrome_weight": float("nan"),
            "max_syndrome_weight": float("nan"),
        }

    return {
        "mean_syndrome_weight": float(np.mean(selected)),
        "median_syndrome_weight": float(np.median(selected)),
        "min_syndrome_weight": float(np.min(selected)),
        "max_syndrome_weight": float(np.max(selected)),
    }


def analyse_pair(
    seed: int,
    run_a: CRNRun,
    run_b: CRNRun,
) -> list[dict[str, object]]:
    # Perform shot-by-shot paired analysis of two decoder configurations

    detections_a = np.load(run_a.detections_file)
    detections_b = np.load(run_b.detections_file)

    if not np.array_equal(detections_a, detections_b):
        raise RuntimeError(
            f"Cannot compare seed {seed}: detection arrays differ."
        )

    correct_a = np.load(run_a.shot_correct_file).astype(bool)
    correct_b = np.load(run_b.shot_correct_file).astype(bool)

    if correct_a.shape != correct_b.shape:
        raise RuntimeError(
            f"Correctness arrays have different shapes: "
            f"{correct_a.shape} vs {correct_b.shape}."
        )

    if detections_a.shape[0] != correct_a.shape[0]:
        raise RuntimeError(
            "Number of detection samples does not match number of "
            "correctness samples."
        )

    weights = syndrome_weights(detections_a)

    categories = {
        "both_succeed": correct_a & correct_b,
        "a_succeeds_b_fails": correct_a & ~correct_b,
        "a_fails_b_succeeds": ~correct_a & correct_b,
        "both_fail": ~correct_a & ~correct_b,
    }

    rows: list[dict[str, object]] = []

    print()
    print(
        f"Seed {seed}: "
        f"beam {run_a.det_beam}/pq {run_a.pqlimit} "
        f"vs beam {run_b.det_beam}/pq {run_b.pqlimit}"
    )
    print("-" * 80)

    for category, mask in categories.items():
        count = int(mask.sum())
        stats = describe_weights(weights, mask)

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
                "beam_b": run_b.det_beam,
                "pqlimit_b": run_b.pqlimit,
                "category": category,
                "count": count,
                **stats,
            }
        )

    failures_a = int((~correct_a).sum())
    failures_b = int((~correct_b).sum())

    rescued_by_b = int((~correct_a & correct_b).sum())
    regressions_b = int((correct_a & ~correct_b).sum())

    print()
    print(f"A failures             : {failures_a}")
    print(f"B failures             : {failures_b}")
    print(f"Shots rescued by B     : {rescued_by_b}")
    print(f"Shots regressed under B: {regressions_b}")
    print(
        f"Net failure improvement: "
        f"{failures_a - failures_b}"
    )

    return rows


def write_results(
    rows: list[dict[str, object]],
    output: Path,
) -> None:
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "seed",
        "beam_a",
        "pqlimit_a",
        "beam_b",
        "pqlimit_b",
        "category",
        "count",
        "mean_syndrome_weight",
        "median_syndrome_weight",
        "min_syndrome_weight",
        "max_syndrome_weight",
    ]

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved paired analysis to: {output}")


def main() -> None:
    args = parse_args()

    runs = load_runs(args.run_dir)

    if not runs:
        raise RuntimeError(
            f"No CRN runs found under {args.run_dir}."
        )

    print_run_summary(runs)

    grouped = group_by_seed(runs)

    print("\nValidating CRN samples")
    print("======================")

    for seed, seed_runs in sorted(grouped.items()):
        validate_crn_group(
            seed,
            seed_runs,
        )

    all_rows: list[dict[str, object]] = []

    print("\nPaired decoder analysis")
    print("=======================")

    for seed, seed_runs in sorted(grouped.items()):
        configurations = {
            (run.det_beam, run.pqlimit): run
            for run in seed_runs
        }

        # Optuna pqlimit x beam study
        desired_pairs = [
            ((10, 75000), (15, 20000)),
            ((15, 20000), (25, 50000)),
            ((10, 75000), (25, 50000)),
        ]

        for config_a, config_b in desired_pairs:
            if config_a not in configurations:
                print(
                    f"Skipping seed {seed}: "
                    f"missing configuration {config_a}."
                )
                continue

            if config_b not in configurations:
                print(
                    f"Skipping seed {seed}: "
                    f"missing configuration {config_b}."
                )
                continue

            rows = analyse_pair(
                seed=seed,
                run_a=configurations[config_a],
                run_b=configurations[config_b],
            )

            all_rows.extend(rows)

    write_results(
        all_rows,
        args.output,
    )


if __name__ == "__main__":
    main()
