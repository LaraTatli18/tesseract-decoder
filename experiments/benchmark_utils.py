from __future__ import annotations

import csv
import math
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass(frozen=True)
class DecodeStatistics:
    correct: int
    low_confidence: int
    syndrome_weights: list[int]
    correction_sizes: list[int]
    correction_costs: list[float]
    shot_correct: np.ndarray
    decode_time_seconds: float


@dataclass(frozen=True)
class BenchmarkResult:  # Schema for result from ONE CIRCUIT
    stim_file: str
    rounds: int  # meta
    distance: int  # meta
    physical_error_rate: float  # meta
    noise_model: str  # meta
    code: str  # meta
    num_qubits: int  # meta
    gates: str  # meta
    n_shots: int
    num_detectors: int
    num_observables: int
    dem_entries: int
    correct_decodings: int
    logical_failures: int
    decoder_accuracy: float
    logical_error_rate: float
    logical_error_rate_per_round: float
    low_confidence_rate: float
    mean_syndrome_weight: float
    mean_correction_size: float
    mean_correction_cost: float
    logical_error_rate_ci_low: float  # lower bound of confidence interval on logical error rate
    logical_error_rate_ci_high: float  # higher bound of confidence interval on logical error rate
    decode_time_seconds: float
    shots_per_second: float
    decode_mode: str
    workers: int
    # AUTOTUNING PARAMS:
    det_beam: int
    beam_climbing: bool
    merge_errors: bool
    pqlimit: int
    det_penalty: float
    sparsify_errors: bool
    sparsify_base_degree: int
    sparsify_max_degree: int
    sparsify_reactivate_limit: int
    # optional bb code metadata:
    nkd: str | None = None
    is_coloured: bool | None = None
    a_poly: str | None = None
    b_poly: str | None = None


def parse_stim_filename(stim_path: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    stem = stim_path.stem

    items: list[str] = []
    current: list[str] = []
    bracket_depth = 0

    for char in stem:
        if char in "[({":
            bracket_depth += 1
        elif char in "])}":
            bracket_depth -= 1
            if bracket_depth < 0:
                raise ValueError(
                    f"Unbalanced brackets in Stim filename: {stim_path.name}"
                )

        if char == "," and bracket_depth == 0:
            items.append("".join(current))
            current = []
        else:
            current.append(char)

    if bracket_depth != 0:
        raise ValueError(
            f"Unbalanced brackets in Stim filename: {stim_path.name}"
        )

    if current:
        items.append("".join(current))

    for item in items:
        if "=" not in item:
            raise ValueError(
                f"Malformed metadata field {item!r} "
                f"in Stim filename: {stim_path.name}"
            )

        key, value = item.split("=", maxsplit=1)
        metadata[key] = value

    return metadata


def _print_histogram(title: str, values: list[int]) -> None:
    print(title)
    print("-" * len(title))

    if not values:
        print("No data")
        print()
        return

    counts: dict[int, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1

    for key in sorted(counts):
        print(f"{key:>3} : {counts[key]}")

    print()


def _write_summary_csv(output_csv: Path, result: BenchmarkResult) -> None:
    """Append one row of summary statistics to a CSV file."""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_csv.exists()

    row = asdict(result)

    with output_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _wilson_interval(successes: int, trials: int,
                     z: float = 1.6448536269514722) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""

    if trials == 0:
        return 0.0, 0.0

    p = successes / trials
    denom = 1.0 + z * z / trials
    center = (p + z * z / (2 * trials)) / denom
    radius = z * math.sqrt(
        (p * (1 - p) + z * z / (4 * trials)) / trials) / denom
    return max(0.0, center - radius), min(1.0, center + radius)


def _per_round_error_rate(logical_error_rate: float, rounds: int) -> float:
    if rounds <= 0:
        return float("nan")
    return 0.5 * (1 - (1 - 2 * logical_error_rate) ** (1 / rounds))


def _select_benchmark_files(
    stim_dir: Path,
    basis: str,
    distances: set[int],
    p_values: set[float],
) -> list[Path]:
    stim_files: list[Path] = []

    for f in sorted(stim_dir.glob("*.stim")):
        metadata = parse_stim_filename(f)
        if metadata["c"] != basis:
            continue
        if int(metadata["d"]) not in distances:
            continue
        if float(metadata["p"]) not in p_values:
            continue
        stim_files.append(f)

    stim_files.sort(
        key=lambda f: (
            int(parse_stim_filename(f)["d"]),
            float(parse_stim_filename(f)["p"]),
            parse_stim_filename(f)["c"],
            f.name,
        )
    )
    return stim_files
