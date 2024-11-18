""" Python program that implemenets the Fast Fourier Transform (FFT) algorithm. """

import numpy as np
import sys
from fftQuery import fftQuery, fftQueryParsingError
from FastMode import FastMode
from DenoiseMode import DenoiseMode
from CompressMode import CompressMode
from PlottingMode import PlottingMode
from QueryMode import Mode
import cv2
from DiscreteFourierTransform import dft_naive_2D, dft_naive_1D


def main():
    try:
        query = fftQuery.parseArguments(sys.argv[1:])        
    except fftQueryParsingError as error:
        print("ERROR\tIncorrect input syntax: " + str(error))
        exit()
    print(query)
    BGR_array = fftQuery.convert_image_to_numpy_array(query.image_name)
    print(dft_naive_2D(dft_naive_2D(np.array([[1, 6,  2],[32,51,61]])), True))
    exit()
    cv2.imwrite('../Results/output.png', dft_naive_2D(BGR_array))

    if query.mode == Mode.FAST:
        FastMode(np.array(BGR_array, dtype=np.float64))
    elif query.mode == Mode.DENOISE:
        DenoiseMode(np.array(BGR_array, dtype=np.float64))
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
