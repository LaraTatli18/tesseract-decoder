from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import csv
import math
import time

import numpy as np
import stim

from tesseract_decoder import tesseract

ROOT = Path(__file__).resolve().parents[1]
#ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class BenchmarkResult: # schema for results from ONE CIRCUIT. want to scale to multiple workers returning this in parallel version
    stim_file: str
    rounds: int
    distance: int
    physical_error_rate: float
    noise_model: str
    code: str
    num_qubits: int
    gates: str
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
    logical_error_rate_ci_low: float
    logical_error_rate_ci_high: float
    decode_time_seconds: float
    shots_per_second: float


def parse_stim_filename(stim_path: Path) -> dict[str, str]:
    "Extract metadata from filenames like r=3,d=3,p=0.001,noise=si1000,c=surface_code_X,q=17,gates=cz.stim "

    metadata: dict[str, str] = {}
    stem = stim_path.stem
    for item in stem.split(","):
        key, value = item.split("=", maxsplit=1)
        metadata[key] = value
    return metadata


def _print_histogram(title: str, values: list[int]) -> None:
    """Print a simple integer histogram sorted by bin."""

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


def _write_summary_csv(output_csv: Path, row: dict[str, Any]) -> None:
    """Append one row of summary statistics to a CSV file."""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_csv.exists()

    with output_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _wilson_interval(successes: int, trials: int, z: float = 1.6448536269514722) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    The default z corresponds to a two-sided 90% confidence interval.
    """

    if trials == 0:
        return 0.0, 0.0

    p = successes / trials
    denom = 1.0 + z * z / trials
    center = (p + z * z / (2 * trials)) / denom
    radius = z * math.sqrt((p * (1 - p) + z * z / (4 * trials)) / trials) / denom
    return max(0.0, center - radius), min(1.0, center + radius)


def _per_round_error_rate(logical_error_rate: float, rounds: int) -> float:
    """Convert shot error rate into the per-round rate used in the paper."""

    if rounds <= 0:
        return float("nan")
    return 0.5 * (1 - (1 - 2 * logical_error_rate) ** (1 / rounds))

def _select_benchmark_files(
    stim_dir: Path,
    basis: str,
    p_values: set[str],
) -> list[Path]:
    """Select and sort the circuits we want to benchmark."""

    stim_files = [
        f
        for f in sorted(stim_dir.glob("*.stim"))
        if parse_stim_filename(f)["c"] == basis
        and parse_stim_filename(f)["p"] in p_values
    ]
    stim_files.sort(
        key=lambda f: (
            int(parse_stim_filename(f)["d"]),
            float(parse_stim_filename(f)["p"]),
        )
    )
    return stim_files


def analyse_many_shots(stim_path: Path, n_shots: int, verbose_histograms: bool, print_every: int, output_csv: Path | None) -> BenchmarkResult:
    print("=" * 100)
    print(f"Analysing shot: {stim_path.name}")
    print("=" * 100)

    circuit = stim.Circuit.from_file(stim_path)
    dem = circuit.detector_error_model(decompose_errors=True)
    dem_entries = list(dem)
    metadata = parse_stim_filename(stim_path)

    config = tesseract.TesseractConfig(dem=dem)
    decoder = config.compile_decoder()

    print(f"Detector count         : {decoder.num_detectors}")
    print(f"Observable count       : {decoder.num_observables}")
    print()

    sampler = circuit.compile_detector_sampler()
    detections, observables = sampler.sample(shots=n_shots, separate_observables=True)

    correct = 0
    low_confidence = 0
    syndrome_weights: list[int] = []
    correction_sizes: list[int] = []
    correction_costs: list[float] = []

    start_time = time.perf_counter()

    for shot_index, (syndrome, truth) in enumerate(zip(detections, observables),
                                                   start=1):
        predicted_errors = decoder.decode_to_errors(syndrome)
        predicted_obs = decoder.get_observables_from_errors(predicted_errors)
        cost = decoder.cost_from_errors(predicted_errors)
        was_low_confidence = bool(decoder.low_confidence_flag)

        syndrome_weights.append(int(np.count_nonzero(syndrome)))
        correction_sizes.append(len(predicted_errors))
        correction_costs.append(float(cost))

        if was_low_confidence:
            low_confidence += 1

        if np.array_equal(predicted_obs, truth):
            correct += 1

        if print_every > 0 and shot_index % print_every == 0:
            print(f"  processed {shot_index}/{n_shots} shots")

    decode_time_seconds = time.perf_counter() - start_time
    shots_per_second = n_shots / decode_time_seconds if decode_time_seconds > 0 else float(
        "inf")

    logical_failures = n_shots - correct
    decoder_accuracy = correct / n_shots
    logical_error_rate = logical_failures / n_shots
    low_confidence_rate = low_confidence / n_shots
    mean_syndrome_weight = float(
        np.mean(syndrome_weights)) if syndrome_weights else 0.0
    mean_correction_size = float(
        np.mean(correction_sizes)) if correction_sizes else 0.0
    mean_correction_cost = float(
        np.mean(correction_costs)) if correction_costs else 0.0
    rounds = int(metadata["r"])
    logical_error_rate_per_round = _per_round_error_rate(logical_error_rate,
                                                         rounds)
    ci_low, ci_high = _wilson_interval(logical_failures, n_shots)

    print()
    print("=" * 100)
    print("Summary")
    print("=" * 100)
    print(f"Shots analysed          : {n_shots}")
    print(f"Correct decodings       : {correct}")
    print(f"Logical failures        : {logical_failures}")
    print()
    print(f"Decoder accuracy        : {decoder_accuracy:.6f}")
    print(f"Logical error rate      : {logical_error_rate:.6f}")
    print(f"Logical error rate/round: {logical_error_rate_per_round:.6e}")
    print(f"90% CI (shot error)     : [{ci_low:.6f}, {ci_high:.6f}]")
    print(f"Low confidence rate     : {low_confidence_rate:.6f}")
    print(f"Decode time (seconds)    : {decode_time_seconds:.3f}")
    print(f"Shots per second         : {shots_per_second:.2f}")
    print()
    print(f"Mean syndrome weight    : {mean_syndrome_weight:.3f}")
    print(f"Mean correction size    : {mean_correction_size:.3f}")
    print(f"Mean correction cost    : {mean_correction_cost:.3f}")

    if verbose_histograms:
        _print_histogram("Syndrome weight histogram", syndrome_weights)
        _print_histogram("Correction size histogram", correction_sizes)

    result = BenchmarkResult(
        stim_file=stim_path.name,
        rounds=rounds,
        distance=int(metadata["d"]),
        physical_error_rate=float(metadata["p"]),
        noise_model=metadata["noise"],
        code=metadata["c"],
        num_qubits=int(metadata["q"]),
        gates=metadata["gates"],
        n_shots=n_shots,
        num_detectors=decoder.num_detectors,
        num_observables=decoder.num_observables,
        dem_entries=len(dem_entries),
        correct_decodings=correct,
        logical_failures=logical_failures,
        decoder_accuracy=decoder_accuracy,
        logical_error_rate=logical_error_rate,
        logical_error_rate_per_round=logical_error_rate_per_round,
        low_confidence_rate=low_confidence_rate,
        mean_syndrome_weight=mean_syndrome_weight,
        mean_correction_size=mean_correction_size,
        mean_correction_cost=mean_correction_cost,
        logical_error_rate_ci_low=ci_low,
        logical_error_rate_ci_high=ci_high,
        decode_time_seconds=decode_time_seconds,
        shots_per_second=shots_per_second,
    )

    if output_csv is not None:
        _write_summary_csv(output_csv, asdict(result))
        print(f"Saved summary to : {output_csv}")

    return result


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--n-shots",
        type=int,
        default=10000,
        help="Number of shots per circuit.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit number of stim files to analyse.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / "experiments" / "analyse_many_shots.csv",
        help="CSV file to append summary rows to.",
    )
    parser.add_argument(
        "--stim-dir",
        type=Path,
        default=ROOT / "testdata" / "surfacecodes",
        help="Directory containing .stim files.",
    )
    parser.add_argument(
        "--verbose-histograms",
        action="store_true",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=1000,
    )
    args = parser.parse_args()

    # USE THIS FOR DEBUGGING:
    # stim_files = sorted(args.stim_dir.glob("*.stim"))
    # if args.max_files is not None:
    #     stim_files = stim_files[: args.max_files]

    # using this for reproducing paper fig
    stim_files = [
        f for f in sorted(args.stim_dir.glob("*.stim"))
        if parse_stim_filename(f)["c"] == "surface_code_X"
           and parse_stim_filename(f)["p"] in {"0.0005", "0.001", "0.002"}
    ]
    stim_files.sort(
        key=lambda f: (
            int(parse_stim_filename(f)["d"]),
            float(parse_stim_filename(f)["p"]),
        )
    )
    if args.max_files is not None:
        stim_files = stim_files[: args.max_files]

    print(f"Found {len(stim_files)} circuits.\n")

    for stim_file in stim_files:
        analyse_many_shots(stim_file, n_shots=args.n_shots, output_csv=args.output_csv, verbose_histograms=args.verbose_histograms, print_every=args.print_every)
        print()
