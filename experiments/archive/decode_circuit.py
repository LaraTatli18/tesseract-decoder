from pathlib import Path

import numpy as np
import stim

from tesseract_decoder import tesseract

ROOT = Path(__file__).resolve().parents[1]

def run_one(stim_path, shots):
    circuit = stim.Circuit.from_file(str(stim_path))
    dem = circuit.detector_error_model(decompose_errors=True)

    config = tesseract.TesseractConfig(dem=dem)
    decoder = config.compile_decoder()

    sampler = circuit.compile_detector_sampler()
    detections, observables = sampler.sample(
        shots=shots,
        separate_observables=True,
    )

    num_errors = 0
    num_low_confidence = 0

    for k in range(shots):
        predicted = decoder.decode(detections[k])

        if decoder.low_confidence_flag:
            num_low_confidence += 1

        if not np.array_equal(predicted, observables[k]):
            num_errors += 1

    print(f"shots      = {shots}")
    print(f"num_errors = {num_errors}")
    print(f"num_low_confidence = {num_low_confidence}")
    print(f"error_rate = {num_errors / shots:.4f}")

    print("stim circuit: ", stim_path.name)

    # Find what the Python API provides:
    # print("dir(tesseract): ", dir(tesseract))
    # print("dir(decoder): ", dir(decoder))


if __name__ == "__main__":
    stim_file = (
        ROOT
        / "testdata"
        / "surfacecodes"
        / "r=9,d=9,p=0.002,noise=si1000,c=surface_code_X,q=161,gates=cz.stim"
    )
    run_one(stim_file, shots=1000)
