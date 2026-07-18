from pathlib import Path

import numpy as np
import stim

from tesseract_decoder import tesseract

ROOT = Path(__file__).resolve().parents[1]


def analyse_one_shot(stim_path):
    print("=" * 100)
    print(f"Analysing shot: {stim_path.name}")
    print("=" * 100)

    # Build circuit and detector error model

    circuit = stim.Circuit.from_file(stim_path)
    dem = circuit.detector_error_model(decompose_errors=True) # Instead of 1 complicated hyperedge, decompose into several small edges
    dem_entries = list(dem) # Which fault mechanisms Tesseract inferred for this shot


    print("Detector Error Model: ")
    print("--------------------")
    print(f"Type                : {type(dem).__name__}")
    print(f"Number of entries   : {len(dem_entries)}")
    print()
    print()

    # Compile decoder

    config = tesseract.TesseractConfig(dem=dem)
    decoder = config.compile_decoder()

    print("Decoder: ")
    print("-------")
    print(f"Detectors           : {decoder.num_detectors}")
    print(f"Observables         : {decoder.num_observables}")
    print()

    # Sample one shot

    sampler = circuit.compile_detector_sampler()

    detections, observables = sampler.sample(
        shots=1,
        separate_observables=True,
    )

    syndrome = detections[0] # A boolean numpy array that equals the number of detectors
    truth = observables[0]

    fired_detectors = np.flatnonzero(syndrome)

    print("Observed syndrome: ")
    print("-----------------")
    print(f"Syndrome weight     : {len(fired_detectors)}")

    if len(fired_detectors) == 0:
        print("No detector events")
    else:
        print("Detector events")
        for d in fired_detectors:
            print(f"  D{d}")

    print()

    # Decode

    predicted_errors = decoder.decode_to_errors(syndrome)
    predicted_obs = decoder.get_observables_from_errors(predicted_errors)
    cost = decoder.cost_from_errors(predicted_errors)

    print("Predicted correction: ")
    print("--------------------")

    if predicted_errors:
        print(
            f"Correction size     : {len(predicted_errors)} DEM "
            f"{'entry' if len(predicted_errors) == 1 else 'entries'}"
        )

        for idx in predicted_errors:
            print(f"  Entry {idx}")
            print(f"    {dem_entries[idx]}")

    else:
        print("Correction size     : 0 DEM entries")
        print("None")

    print()

    print(f"Correction cost     : {cost:.6f}")
    print(f"Low confidence      : {decoder.low_confidence_flag}")
    print()

    # Logical prediction

    print("Logical observable: ")
    print("------------------")
    print(f"Predicted          : {predicted_obs}")
    print(f"Ground truth       : {truth}")
    print(f"Decoder correct    : {np.array_equal(predicted_obs, truth)}")
    print()

    # API sanity checks

    print("API checks: ")
    print("----------")
    print(f"decode() result    : {decoder.decode(syndrome)}")
    print(f"DEM entries used   : {len(predicted_errors)}")


if __name__ == "__main__":

    stim_file = (
        ROOT
        / "testdata"
        / "surfacecodes"
        / "r=3,d=3,p=0.001,noise=si1000,c=surface_code_X,q=17,gates=cz.stim"
    )

    analyse_one_shot(stim_file)
