""" Python program that implemenets the Fast Fourier Transform (FFT) algorithm. """

import numpy as np
import sys
from fftQuery import fftQuery, fftQueryParsingError
from FastMode import FastMode
from DenoiseMode import DenoiseMode
import DiscreteFourierTransform2D as dft2d
from QueryMode import Mode



def main():
    try:
        query = fftQuery.parseArguments(sys.argv[1:])
        image = query.get_image()

    except fftQueryParsingError as error:
        print("ERROR\tIncorrect input syntax: " + str(error))
        exit()

    case = query.get_mode()

    if case == Mode.FAST:
        FastMode(np.array(image, dtype=np.float64))

    elif case == Mode.DENOISE:
        DenoiseMode(np.array(image, dtype=np.float64))
        pass
    elif case == Mode.COMPRESS:

        pass
    elif case == Mode.PLOT_RUNTIME:

        pass
    else:
        print("ERROR\tUnknown mode: " + case)
        exit()



# For when the program is invoked from the command line (stdin)
if __name__ == "__main__":
    main()
