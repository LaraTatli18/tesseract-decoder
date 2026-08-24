from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import stim


CIRCUIT = Path(
    "testdata/surfacecodes/"
    "r=11,d=11,p=0.002,noise=si1000,"
    "c=surface_code_X,q=241,gates=cz.stim"
)

DETECTIONS = Path(
    "experiments/runs/tesseract_default_crn_validation/"
    "2026-08-16_114436_892213_surface_code_X_d11_p0.002_250000shots_1workers_32threads_"
    "beam5_pq200000_bc0_me1_sp0__crnseed4122171318702187655/"
    "samples/"
    "r=11,d=11,p=0.002,noise=si1000,c=surface_code_X,q=241,gates=cz_detections.npy"
)

SHOT_INDEX = 88401

OUTPUT = Path(
    "viz/crn_rescued_shot_88401.json"
)


def main() -> None:
    circuit = stim.Circuit.from_file(
        str(CIRCUIT)
    )

    detector_coords = (
        circuit.get_detector_coordinates()
    )

    detections = np.load(
        DETECTIONS
    )

    syndrome = detections[
        SHOT_INDEX
    ].astype(bool)

    if len(detector_coords) != len(syndrome):
        raise RuntimeError(
            f"Detector-coordinate count "
            f"{len(detector_coords)} does not match "
            f"syndrome length {len(syndrome)}."
        )

    activated = np.flatnonzero(
        syndrome
    ).tolist()

    data = {
        "detectorCoords": {
            str(i): list(coord)
            for i, coord in detector_coords.items()
        },
        "errorCoords": {},
        "errorToDetectors": {},
        "frames": [
            {
                "activated": activated,
                "activated_errors": [],
            }
        ],
    }

    OUTPUT.write_text(
        json.dumps(data, indent=2)
    )

    print(
        f"Saved: {OUTPUT}"
    )

    print(
        f"Syndrome weight: "
        f"{len(activated)}"
    )

    print(
        f"Activated detectors: "
        f"{activated}"
    )


if __name__ == "__main__":
    main()
