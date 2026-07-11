from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import csv
import stim

from tesseract_decoder import tesseract

ROOT = Path(__file__).resolve().parents[1]

def parse_stim_filename(stim_path: Path) -> dict:

    # Extract metadata from filenames like r=3,d=3,p=0.001,noise=si1000,c=surface_code_X,q=17,gates=cz.stim

    metadata = {}

    stem = stim_path.stem

    for item in stem.split(","):
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

    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1

    for key in sorted(counts):
        print(f"{key:>3} : {counts[key]}")

    print()

def _write_summary_csv(output_csv: Path, row: dict[str, object]) -> None:
    # Append one row of summary statistics to a CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    file_exists = output_csv.exists()
    with output_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def analyse_many_shots(stim_path, N_shots, output_csv):
    print("=" * 100)
    print(f"Analysing shot: {stim_path.name}")
    print("=" * 100)

    # ------------------------------------------------------------------
    # Build circuit and detector error model
    # ------------------------------------------------------------------

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

    detections, observables = sampler.sample(
        shots=N_shots,
        separate_observables=True,
    )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    correct = 0
    low_confidence = 0

    syndrome_weights = []
    correction_sizes = []
    correction_costs = []

    # ------------------------------------------------------------------
    # Decode every shot
    # ------------------------------------------------------------------

    for syndrome, truth in zip(detections, observables):

        predicted_errors = decoder.decode_to_errors(syndrome)
        predicted_obs = decoder.get_observables_from_errors(predicted_errors)
        cost = decoder.cost_from_errors(predicted_errors)

        # Record statistics

        syndrome_weights.append(np.count_nonzero(syndrome))
        correction_sizes.append(len(predicted_errors))
        correction_costs.append(cost)

        if decoder.low_confidence_flag:
            low_confidence += 1

        if np.array_equal(predicted_obs, truth):
            correct += 1

        logical_failures = N_shots - correct
        decoder_accuracy = correct / N_shots
        logical_error_rate = logical_failures / N_shots
        low_confidence_rate = low_confidence / N_shots
        mean_syndrome_weight = float(np.mean(syndrome_weights))
        mean_correction_size = float(np.mean(correction_sizes))
        mean_correction_cost = float(np.mean(correction_costs))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    print()
    print("=" * 100)
    print("Summary")
    print("=" * 100)

    print(f"Shots analysed          : {N_shots}")
    print(f"Correct decodings       : {correct}")
    print(f"Logical failures        : {N_shots - correct}")

    print()

    print(f"Decoder accuracy        : {correct / N_shots:.6f}")
    print(f"Logical error rate      : {(N_shots - correct) / N_shots:.6f}")
    print(f"Low confidence rate     : {low_confidence / N_shots:.6f}")

    print()

    print(f"Mean syndrome weight    : {np.mean(syndrome_weights):.3f}")
    print(f"Mean correction size    : {np.mean(correction_sizes):.3f}")
    print(f"Mean correction cost    : {np.mean(correction_costs):.3f}")

    _print_histogram("Syndrome weight histogram", syndrome_weights)
    _print_histogram("Correction size histogram", correction_sizes)

    if output_csv is not None:
        row = {
            "stim_file": stim_path.name,

            "rounds": int(metadata["r"]),
            "distance": int(metadata["d"]),
            "physical_error_rate": float(metadata["p"]),
            "noise_model": metadata["noise"],
            "code": metadata["c"],
            "num_qubits": int(metadata["q"]),
            "gates": metadata["gates"],

            "n_shots": N_shots,

            "num_detectors": decoder.num_detectors,
            "num_observables": decoder.num_observables,
            "dem_entries": len(dem_entries),

            "correct_decodings": correct,
            "logical_failures": logical_failures,

            "decoder_accuracy": decoder_accuracy,
            "logical_error_rate": logical_error_rate,
            "low_confidence_rate": low_confidence_rate,

            "mean_syndrome_weight": mean_syndrome_weight,
            "mean_correction_size": mean_correction_size,
            "mean_correction_cost": mean_correction_cost,
        }

        _write_summary_csv(output_csv, row)
        print(f"Saved summary to : {output_csv}")

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
    args = parser.parse_args()

    surfacecode_dir = ROOT / "testdata" / "surfacecodes"
    stim_files = sorted(surfacecode_dir.glob("*.stim"))

    if args.max_files is not None:
        stim_files = stim_files[: args.max_files]

    print(f"Found {len(stim_files)} circuits.\n")

    for stim_file in stim_files:
        analyse_many_shots(
            stim_file,
            N_shots=args.n_shots,
            output_csv=args.output_csv,
        )
        print()

# if __name__ == "__main__":
#
#     surfacecode_dir = ROOT / "testdata" / "surfacecodes"
#
#     stim_files = sorted(surfacecode_dir.glob("*.stim"))
#
#     print(f"Found {len(stim_files)} circuits.\n")
#
#     stim_files = stim_files[:3]
#     for stim_file in stim_files:
#         analyse_many_shots(
#             stim_file,
#             N_shots=10000,
#             output_csv=ROOT / "experiments" / "analyse_many_shots.csv",
#         )
#
#         print()
