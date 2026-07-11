from tesseract_decoder import tesseract

import stim

import numpy as np

dem = stim.DetectorErrorModel("""

    error(0.1) D0 D1 L0

    error(0.2) D1 D2 L1

    detector(0,0,0) D0

    detector(1,0,0) D1

    detector(2,0,0) D2

""")

config = tesseract.TesseractConfig(

    dem=dem,

    det_beam=50

)

decoder = config.compile_decoder()

syndrome = np.array([1,1,0], dtype=bool)

print(decoder.decode(syndrome))

