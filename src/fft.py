""" Python program that implemenets the Fast Fourier Transform (FFT) algorithm. """

import numpy as np
import sys
from fftQuery import fftQuery, fftQueryParsingError
from FastMode import FastMode
from DenoiseMode import DenoiseMode
from CompressMode import CompressMode
from PlottingMode import PlottingMode
from QueryMode import Mode



def main():
    try:
        query = fftQuery.parseArguments(sys.argv[1:])
        
        image = fftQuery.convert_image_to_numpy_array(query.image_name)

    except fftQueryParsingError as error:
        print("ERROR\tIncorrect input syntax: " + str(error))
        exit()

    case = query.mode

    if case == Mode.FAST:
        FastMode(np.array(image, dtype=np.float64))
    elif case == Mode.DENOISE:
        DenoiseMode(np.array(image, dtype=np.float64))
    elif case == Mode.COMPRESS:
        CompressMode(np.array(image, dtype=np.float64))
    elif case == Mode.PLOT_RUNTIME:
        PlottingMode()
    else:
        print("ERROR\tUnknown mode: " + case)
        exit()


# For when the program is invoked from the command line (stdin)
if __name__ == "__main__":
    main()
