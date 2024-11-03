""" Python program that implemenets the Fast Fourier Transform (FFT) algorithm. """

import numpy as np
import sys
from fftQuery import fftQuery, fftQueryParsingError


def main():
    try:
        query = fftQuery.parseArguments(sys.argv[1:])
        mode = query.get_mode()
        image = query.get_image()

    except fftQueryParsingError as error:
        print("ERROR\tIncorrect input syntax: " + error.value)
        exit()

    pass


# For when the program is invoked from the command line (stdin)
if __name__ == "__main__":
    main()
