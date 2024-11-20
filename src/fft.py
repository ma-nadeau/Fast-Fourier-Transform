""" Python program that implemenets the Fast Fourier Transform (FFT) algorithm. """

import os
import numpy as np
import sys
from DenoiseMode import plot_denoise_mode
from fftQuery import fftQuery, fftQueryParsingError
from FastMode import plot_fast_mode
# from DenoiseMode import DenoiseMode
# from CompressMode import CompressMode
# from PlottingMode import PlottingMode
from QueryMode import Mode
import cv2
import numpy as np


def main():
    try:
        query = fftQuery.parseArguments(sys.argv[1:])        
    except fftQueryParsingError as error:
        print("ERROR\tIncorrect input syntax: " + str(error))
        exit()
    
    original_image = fftQuery.convert_image_to_numpy_array(query.image_name)

    if query.mode == Mode.FAST:
        plot_fast_mode(original_image)
    elif query.mode == Mode.DENOISE:
        plot_denoise_mode(original_image)
    elif query.mode == Mode.COMPRESS:
        CompressMode(np.array(BGR_array, dtype=np.float64))
    elif query.mode == Mode.PLOT_RUNTIME:
        PlottingMode()
    else:
        print("ERROR\tUnknown mode: " + query.mode)
        exit()


# For when the program is invoked from the command line (stdin)
if __name__ == "__main__":
    main()
