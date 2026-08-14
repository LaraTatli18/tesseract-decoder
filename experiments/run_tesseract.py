from __future__ import annotations

from argparse import ArgumentParser, BooleanOptionalAction
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import stim
import time
import hashlib

from tesseract_decoder import tesseract

from experiments.benchmark_utils import (
    BenchmarkResult,
    DecodeStatistics,
    _per_round_error_rate,
    _print_histogram,
    _select_benchmark_files,
    _wilson_interval,
    _write_summary_csv,
    parse_stim_filename,
)

from experiments.run_manifest import (
    make_run_directory,
    build_manifest,
    write_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


# ================================================
# MAIN JOBS:
# 1. FIND .STIM CIRCUITS IN TESSERACT REPO
# 2. RUN THE DECODER ON MANY SAMPLED SHOTS
# 3. SUMMARISE RESULTS AND APPEND TO .CSV FILE
# ================================================

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
    shot_correct = np.zeros(len(detections), dtype=bool)

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

        shot_correct[shot_index - 1] = np.array_equal(predicted_obs, truth)

        if shot_correct[shot_index - 1]:
            correct += 1

        if print_every > 0 and shot_index % print_every == 0:
            print(f"  processed {shot_index}/{len(detections)} shots")

    return DecodeStatistics(
        correct=correct,
        low_confidence=low_confidence,
        syndrome_weights=syndrome_weights,
        correction_sizes=correction_sizes,
        correction_costs=correction_costs,
        shot_correct=shot_correct,
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

    shot_correct = np.all(predicted_obs == observables, axis=1)
    correct = int(np.sum(shot_correct))
    syndrome_weights = np.count_nonzero(detections, axis=1).tolist()

    return DecodeStatistics(
        correct=correct,
        low_confidence=0,
        syndrome_weights=syndrome_weights,
        correction_sizes=[],
        correction_costs=[],
        shot_correct=shot_correct,
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
                        det_penalty: float,
                        sparsify_errors: bool,
                        sparsify_base_degree: int,
                        sparsify_max_degree: int,
                        sparsify_reactivate_limit: int,
                        crn: bool,
                        crn_seed: int,
                        save_samples: bool,
                        samples_dir: Path | None,
                        create_visualization: bool) -> BenchmarkResult:
    print("=" * 100)
    print(f"Analysing circuit: {stim_path.name}")
    print("=" * 100)

    circuit = stim.Circuit.from_file(stim_path)
    dem = circuit.detector_error_model(decompose_errors=True, ignore_decomposition_failures=True)
    dem_entries = sum(1 for _ in dem)
    metadata = parse_stim_filename(stim_path)

    config = tesseract.TesseractConfig(dem=dem,
                                       det_beam=det_beam,
                                       beam_climbing=beam_climbing,
                                       verbose=False,
                                       merge_errors=merge_errors,
                                       pqlimit=pqlimit,
                                       det_penalty=det_penalty,
                                       sparsify_errors=sparsify_errors,
                                       sparsify_base_degree=sparsify_base_degree,
                                       sparsify_max_degree=sparsify_max_degree,
                                       sparsify_reactivate_limit=sparsify_reactivate_limit,
                                       create_visualization=create_visualization
                                       )

    decoder = config.compile_decoder()

    print(f"Detector count         : {decoder.num_detectors}")
    print(f"Observable count       : {decoder.num_observables}")
    print(f"Requested reactivation limit: {sparsify_reactivate_limit}") # using this because TesseractConfig doesn't currently expose a sparsify_reactive_limit attribute to the Python bindings
    print()

    if crn:
        # Derive a deterministic per-circuit seed from the user-supplied CRN seed.
        # This ensures that different decoder configurations applied to the same
        # circuit receive identical sampled detector/observable data, while
        # different circuits do not accidentally reuse the same random stream.
        seed_material = f"{crn_seed}:{stim_path.name}".encode("utf-8")
        circuit_seed = int.from_bytes(
            hashlib.sha256(seed_material).digest()[:8],
            byteorder="little",
            signed=False,
        )
        sampler = circuit.compile_detector_sampler(seed=circuit_seed)
        print(
            f"CRN sampling enabled    : "
            f"seed={crn_seed}, circuit_seed={circuit_seed}"
        )
    else:
        # No CRN enabled; just using normal Monte Carlo sampling.
        sampler = circuit.compile_detector_sampler()

    detections, observables = sampler.sample(
        shots=n_shots,
        separate_observables=True,
    )

    if save_samples:
        if samples_dir is None:
            raise ValueError(
                "samples_dir must be provided when save_samples=True"
            )

        samples_dir.mkdir(parents=True, exist_ok=True)
        sample_stem = stim_path.stem

        detections_path = (
            samples_dir / f"{sample_stem}_detections.npy"
        )

        observables_path = (
            samples_dir / f"{sample_stem}_observables.npy"
        )

        np.save(detections_path, detections)
        np.save(observables_path, observables)

        print(f"Saved detections        : {detections_path}")
        print(f"Saved observables       : {observables_path}")

    if decode_mode == "batch":
        stats = _analyse_batch_shots(decoder, detections, observables, threads)
    elif decode_mode == "single":
        stats = _analyse_single_shots(decoder, detections, observables,
                                      print_every)
    else:
        raise (ValueError(f"Unknown decode mode: {decode_mode}"))

    shot_correct = stats.shot_correct

    if save_samples:
        if samples_dir is None:
            raise ValueError(
                "samples_dir must be provided when save_samples=True"
            )

        samples_dir.mkdir(parents=True, exist_ok=True)
        correctness_path = (
            samples_dir / f"{stim_path.stem}_shot_correct.npy"
        )

        np.save(correctness_path, shot_correct)
        print(f"Saved shot correctness : {correctness_path}")

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
        gates=metadata.get("gates", ""),
        n_shots=n_shots,
        num_detectors=decoder.num_detectors,
        num_observables=decoder.num_observables,
        dem_entries=dem_entries,
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
        det_penalty=det_penalty,
        sparsify_errors=sparsify_errors,
        sparsify_base_degree=sparsify_base_degree,
        sparsify_max_degree=sparsify_max_degree,
        sparsify_reactivate_limit=sparsify_reactivate_limit,
        # Optional BB-code metadata
        nkd=metadata.get("nkd"),
        is_coloured=(
            metadata["iscoloured"].lower() == "true"
            if "iscoloured" in metadata
            else None
        ),
        a_poly=metadata.get("A_poly"),
        b_poly=metadata.get("B_poly"),
    )

    return result


def _worker(task: tuple[Path, int, bool, int, str, int, int, int, bool, bool, int, float, bool, int, int, int, bool, int, bool, Path | None, bool]) -> BenchmarkResult:
    stim_path, n_shots, verbose_histograms, print_every, decode_mode, workers, threads, det_beam, beam_climbing, merge_errors, pqlimit, det_penalty, sparsify_errors, sparsify_base_degree, sparsify_max_degree, sparsify_reactivate_limit, crn, crn_seed, save_samples, samples_dir, create_visualization = task
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
                                 det_penalty,
                                 sparsify_errors,
                                 sparsify_base_degree,
                                 sparsify_max_degree,
                                 sparsify_reactivate_limit,
                                 crn,
                                 crn_seed,
                                 save_samples,
                                 samples_dir,
                                 create_visualization)
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
        "--run-group",
        type=str,
        default=None,
        help="Optional subdirectory under experiments/runs, e.g. for Optuna."
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
        # decode_batch removes the Python loop overhead
        default="batch",
        help="Decode mode."
    )
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
        action=BooleanOptionalAction,
        default=False,
        help="Beam climbing parameter."
    )
    parser.add_argument(
        "--merge-errors",
        action=BooleanOptionalAction,
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
    parser.add_argument(
        "--sparsify_errors",
        action=BooleanOptionalAction,
        default=False,
        help="Enable per-shot sparse error activation."
    )
    parser.add_argument(
        "--sparsify-base-degree",
        type=int,
        default=1,
        help="Maximum detector degree for mandatory errors."
    )
    parser.add_argument(
        "--sparsify-max-degree",
        type=int,
        default=-1,
        help="Maximum detector degree for optional errors that may be reactivated.",
    )
    parser.add_argument(
        "--sparsify-reactivate-limit",
        type=int,
        default=-1,
        help="Maximum number of optional errors to reactivate per shot. Use -1 for auto.",
    )
    parser.add_argument(
        "--crn",
        action=BooleanOptionalAction,
        default=False,
        help=(
            "Enable common-random-number sampling. Runs using the same CRN seed "
            "and Stim circuit receive identical sampled detector/observable shots."
        ),
    )
    parser.add_argument(
        "--crn-seed",
        type=int,
        default=12345,
        help="Base random seed used when --crn is enabled.",
    )
    parser.add_argument(
        "--save-samples",
        action=BooleanOptionalAction,
        default=False,
        help=(
            "Save sampled detector/observable arrays and per-shot correctness "
            "arrays as .npy files in the run directory."
        ),
    )
    parser.add_argument(
        "--create-visualization",
        action=BooleanOptionalAction,
        default=False,
        help=(
            "Enable Tesseract decoder visualization output."
            "Currently marks a diagnostic run intended to be followed by"
            "the C++ Tesseract visualization workflow; actual visualization logs"
            "are generated separately using the src:tesseract executable and "
            "viz/to_json.py."
        ),
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
    run_dir, manifest_path = make_run_directory(args, run_group=args.run_group)
    args.output_csv = run_dir / "results.csv"
    samples_dir = (
        run_dir / "samples"
        if args.save_samples
        else None
    )

    manifest = build_manifest(args, args.output_csv)
    write_manifest(manifest_path, manifest)

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
         args.det_penalty,
         args.sparsify_errors,
         args.sparsify_base_degree,
         args.sparsify_max_degree,
         args.sparsify_reactivate_limit,
         args.crn,
         args.crn_seed,
         args.save_samples,
         samples_dir,
         args.create_visualization)
        for stim_file in stim_files
    ]

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(_worker, tasks):
                _write_summary_csv(args.output_csv, result)

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
                                         det_penalty=args.det_penalty,
                                         sparsify_errors=args.sparsify_errors,
                                         sparsify_base_degree=args.sparsify_base_degree,
                                         sparsify_max_degree=args.sparsify_max_degree,
                                         sparsify_reactivate_limit=args.sparsify_reactivate_limit,
                                         crn=args.crn,
                                         crn_seed=args.crn_seed,
                                         save_samples=args.save_samples,
                                         samples_dir=samples_dir,
                                         create_visualization=args.create_visualization)
            _write_summary_csv(args.output_csv, result)
            print(f"Finished circuit {i}/{len(stim_files)}\n")

        print()
