from __future__ import annotations

import argparse
from argparse import ArgumentParser
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any
import csv
import math
import time
import json
import os
import socket
import subprocess
from datetime import datetime

import numpy as np
import stim

from tesseract_decoder import tesseract

ROOT = Path(__file__).resolve().parents[1]

# ================================================
# MAIN JOBS:
# 1. FIND .STIM CIRCUITS IN TESSERACT REPO
# 2. RUN THE DECODER ON MANY SAMPLED SHOTS
# 3. SUMMARISE RESULTS AND APPEND TO .CSV FILE
# ================================================

@dataclass(frozen=True)
class DecodeStatistics:
    correct: int
    low_confidence: int
    syndrome_weights: list[int]
    correction_sizes: list[int]
    correction_costs: list[float]
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


def parse_stim_filename(stim_path: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    stem = stim_path.stem
    for item in stem.split(","):
        key, value = item.split("=", maxsplit=1)
        metadata[key] = value
    return metadata

def _git_value(args: list[str]) -> str:
    """Return a git value, or 'unknown' if git is unavailable."""
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _git_is_clean() -> bool:
    """Return True if the git working tree is clean."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            text=True,
        )
        return out.strip() == ""
    except Exception:
        return False


def _build_run_name(args: argparse.Namespace) -> str:
    """Create a short, filesystem-friendly run directory name."""
    timestamp = datetime.now().strftime("%Y-%m-%d")
    return (
        f"{timestamp}_"
        f"{args.basis}_"
        f"{args.decode_mode}_"
        f"{args.n_shots}shots_"
        f"{args.workers}workers_"
        f"{args.threads}threads_"
        f"beam{args.det_beam}"
    )


def _make_run_directory(args: argparse.Namespace) -> tuple[Path, Path]:
    """Create the run directory and return (run_dir, manifest_path)."""
    run_name = _build_run_name(args)
    run_dir = ROOT / "experiments" / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    return run_dir, manifest_path


def _build_manifest(args: argparse.Namespace, output_csv: Path) -> dict[str, Any]:
    """Build a JSON-serialisable manifest for this run."""
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "cwd": str(Path.cwd()),
        "git_branch": _git_value(["git", "branch", "--show-current"]),
        "git_commit": _git_value(["git", "rev-parse", "HEAD"]),
        "working_tree_clean": _git_is_clean(),
        "command": ["bazel", "run", "//src/py:run_tesseract", "--", *os.sys.argv[1:]],
        "output_csv": str(output_csv),
        "stim_dir": str(args.stim_dir),
        "basis": args.basis,
        "distances": list(args.distances),
        "p_values": list(args.p_values),
        "n_shots": args.n_shots,
        "decode_mode": args.decode_mode,
        "workers": args.workers,
        "threads": args.threads,
        "det_beam": args.det_beam,
        "beam_climbing": args.beam_climbing,
        "merge_errors": args.merge_errors,
        "pqlimit": args.pqlimit,
        "det_penalty": args.det_penalty,
    }


def _write_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
    """Write the run manifest as pretty JSON."""
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")


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


def _write_summary_csv(output_csv: Path, row: dict[str, Any]) -> None:
    """Append one row of summary statistics to a CSV file."""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_csv.exists()

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


def _analyse_single_shots(
    decoder: tesseract.TesseractDecoder,
    detections: np.ndarray,
    observables: np.ndarray,
    print_every: int,
) -> DecodeStatistics:
    correct = 0
    low_confidence = 0
    syndrome_weights: list[int] = []
    correction_sizes: list[int] = []
    correction_costs: list[float] = []

    # ========================================
    # MAIN DECODING LOOP
    # ========================================

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
            print(f"  processed {shot_index}/{len(detections)} shots")

    return DecodeStatistics(
        correct=correct,
        low_confidence=low_confidence,
        syndrome_weights=syndrome_weights,
        correction_sizes=correction_sizes,
        correction_costs=correction_costs,
        decode_time_seconds=time.perf_counter() - start_time,
    )


def _analyse_batch_shots(
    decoder: tesseract.TesseractDecoder,
    detections: np.ndarray,
    observables: np.ndarray,
    num_threads: int,
) -> DecodeStatistics:
    start_time = time.perf_counter()
    predicted_obs = np.asarray(decoder.decode_batch(detections, num_threads))
    decode_time_seconds = time.perf_counter() - start_time

    if predicted_obs.ndim == 1:
        predicted_obs = predicted_obs[:, np.newaxis]
    if observables.ndim == 1:
        observables = observables[:, np.newaxis]

    correct = int(np.sum(np.all(predicted_obs == observables, axis=1)))
    syndrome_weights = np.count_nonzero(detections, axis=1).tolist()

    return DecodeStatistics(
        correct=correct,
        low_confidence=0,
        syndrome_weights=syndrome_weights,
        correction_sizes=[],
        correction_costs=[],
        decode_time_seconds=decode_time_seconds,
    )


def analyse_one_circuit(stim_path: Path,
                        n_shots: int,
                        verbose_histograms: bool,
                        print_every: int,
                        decode_mode: str,
                        workers: int,
                        threads: int,
                        det_beam: int,
                        beam_climbing: bool,
                        merge_errors: bool,
                        pqlimit: int,
                        det_penalty: float) -> BenchmarkResult:
    print("=" * 100)
    print(f"Analysing circuit: {stim_path.name}")
    print("=" * 100)

    circuit = stim.Circuit.from_file(stim_path)
    dem = circuit.detector_error_model(decompose_errors=True)
    dem_entries = list(dem)
    metadata = parse_stim_filename(stim_path)

    config = tesseract.TesseractConfig(dem=dem,
                                       det_beam=det_beam,
                                       beam_climbing=beam_climbing,
                                       verbose=False,
                                       merge_errors=merge_errors,
                                       pqlimit=pqlimit,
                                       det_penalty=det_penalty)

    decoder = config.compile_decoder()

    print(f"Detector count         : {decoder.num_detectors}")
    print(f"Observable count       : {decoder.num_observables}")
    print()

    sampler = circuit.compile_detector_sampler()
    detections, observables = sampler.sample(shots=n_shots,
                                             separate_observables=True)

    if decode_mode == "batch":
        stats = _analyse_batch_shots(decoder, detections, observables, threads)
    elif decode_mode == "single":
        stats = _analyse_single_shots(decoder, detections, observables,
                                      print_every)
    else:
        raise (ValueError(f"Unknown decode mode: {decode_mode}"))

    correct = stats.correct
    low_confidence = stats.low_confidence
    syndrome_weights = stats.syndrome_weights
    correction_sizes = stats.correction_sizes
    correction_costs = stats.correction_costs
    decode_time_seconds = stats.decode_time_seconds

    # CONSTRUCT BENCHMARK RESULT

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
    print(f"Decode mode      : {decode_mode}")
    print(f"Circuit workers  : {workers}")
    print(f"Decoder threads  : {threads}")
    print()
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
        decode_mode=decode_mode,
        workers=workers,
        det_beam=det_beam,
        beam_climbing=beam_climbing,
        merge_errors=merge_errors,
        pqlimit=pqlimit,
        det_penalty=det_penalty
    )

    return result


def _worker(task: tuple[
    Path, int, bool, int, str, int, int, int, bool, bool, int, float]) -> BenchmarkResult:
    stim_path, n_shots, verbose_histograms, print_every, decode_mode, workers, threads, det_beam, beam_climbing, merge_errors, pqlimit, det_penalty = task
    print(f"Starting circuit: {stim_path.name}")
    result = analyse_one_circuit(stim_path,
                                 n_shots,
                                 verbose_histograms,
                                 print_every,
                                 decode_mode,
                                 workers,
                                 threads,
                                 det_beam,
                                 beam_climbing,
                                 merge_errors,
                                 pqlimit,
                                 det_penalty)
    print(f"Finished circuit: {stim_path.name}")
    return result


# ===========================
# ENGINE
# ===========================

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
        default=None,
        help="CSV file to append summary rows to. If omitted, a descriptive filename is generated automatically."
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
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers."
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of parallel threads for shot multiprocessing."
    )
    parser.add_argument(
        "--decode-mode",
        choices=["single", "batch"],
        default="batch",
        help="Decode mode."
    )
    # decode_batch removes the Python loop overhead
    # Native shot-threading is currently only available through the Tesseract CLI (--threads) - RESOLVED
    parser.add_argument(
        "--basis",
        type=str,
        default="surface_code_X",
        help="Which code basis to test."
    )
    parser.add_argument(
        "--distances",
        type=int,
        nargs="*",
        default=[3, 5, 7, 9, 11])
    parser.add_argument(
        "--p-values",
        type=float,
        nargs="*",
        default=[0.0005, 0.001, 0.002])
    parser.add_argument(
        "--det-beam",
        type=int,
        default=5,
        help="Beam size for deterministic beam search."
    )
    parser.add_argument(
        "--beam-climbing",
        action="store_true",
        help="Beam climbing parameter."
    )
    parser.add_argument(
        "--merge-errors",
        action="store_true",
        default=True,
        help="Merges error channels with identical syndrome patterns before decoding."
    )
    parser.add_argument(
        "--pqlimit",
        type=int,
        default=200000,
        help="An integer that sets a limit on the number of nodes in the priority queue. This can be used to constrain the memory usage of the decoder."
    )
    parser.add_argument(
        "--det-penalty",
        type=float,
        default=0.0,
        help="Penalty parameter that adds a cost for each residual detection event."
    )

    args = parser.parse_args()

    stim_files = _select_benchmark_files(
        stim_dir=args.stim_dir,
        basis=args.basis,
        distances=set(args.distances),
        p_values=set(args.p_values),
    )

    if args.max_files is not None:
        stim_files = stim_files[: args.max_files]

    # =====================================
    # CREATE RUN DIRECTORY/MANIFEST
    # =====================================
    run_dir, manifest_path = _make_run_directory(args)
    args.output_csv = run_dir / "results.csv"

    manifest = _build_manifest(args, args.output_csv)
    _write_manifest(manifest_path, manifest)

    # if args.output_csv is None:
    #     run_dir, manifest_path = _make_run_directory(args)
    #     args.output_csv = run_dir / "results.csv"
    # else:
    #     run_dir = args.output_csv.parent
    #     run_dir.mkdir(parents=True, exist_ok=True)
    #     manifest_path = run_dir / "manifest.json"

    # if output_csv is not None:
        # default_name = (
        #     f"{args.basis}_"
        #     f"{args.decode_mode}_"
        #     f"{args.n_shots}shots_"
        #     f"{args.workers}workers_"
        #     f"{args.threads}threads.csv"
        # )  # ADD AUTOTUNING PARAMS TO NAMES
        # args.output_csv = ROOT / "experiments" / "results" / default_name

    print(f"Run directory     : {run_dir}")
    print(f"Saved results     : {args.output_csv}")
    print(f"Saved manifest    : {manifest_path}")
    print(f"Found {len(stim_files)} circuits.\n")

    tasks = [
        (stim_file,
         args.n_shots,
         args.verbose_histograms,
         args.print_every,
         args.decode_mode,
         args.workers,
         args.threads,
         args.det_beam,
         args.beam_climbing,
         args.merge_errors,
         args.pqlimit,
         args.det_penalty)
        for stim_file in stim_files
    ]

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(_worker, tasks):
                _write_summary_csv(args.output_csv, asdict(result))

    else:
        for i, stim_file in enumerate(stim_files, start=1):
            print(f"Starting circuit {i}/{len(stim_files)}: {stim_file.name}")
            result = analyse_one_circuit(stim_file,
                                         n_shots=args.n_shots,
                                         verbose_histograms=args.verbose_histograms,
                                         print_every=args.print_every,
                                         decode_mode=args.decode_mode,
                                         workers=args.workers,
                                         threads=args.threads,
                                         det_beam=args.det_beam,
                                         beam_climbing=args.beam_climbing,
                                         merge_errors=args.merge_errors,
                                         pqlimit=args.pqlimit,
                                         det_penalty=args.det_penalty)
            _write_summary_csv(args.output_csv, asdict(result))
            print(f"Finished circuit {i}/{len(stim_files)}\n")

        print()
