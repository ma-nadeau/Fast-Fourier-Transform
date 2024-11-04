from typing import List
import os
import cv2
import numpy as np
from enum import Enum
from QueryMode import Mode


class fftQuery:

    utilisation = """ The input should have the following format:
    python fft.py [-m mode] [-i image]
    
    mode (optional):
    - [1] Fast mode (default): Convert image to FFT form and display.
    - [2] Denoise: The image is denoised by applying an FFT, truncating high frequencies, and then displayed.
    - [3] Compress: Compress image and plot.
    - [4] Plot runtime graphs for the report.
    image (optional): 
    - The filename of the image for the DFT (default: 'moonlanding.png').
    """

    def __init__(self, mode: Mode, image_name: str, image: np.ndarray = None) -> None:
        self.mode = mode
        self.image_name = image_name
        self.image = image

    @classmethod
    def parseArguments(cls, argv: List[str]) -> "fftQuery":
        mode = Mode.FAST
        image_name = "../Figures/moonlanding.png"
        image = cls.convert_image_to_numpy_array(image_name)

        # Check that the argument length is valid
        if not 0 <= len(argv) <= 4:
            raise fftQueryParsingError(cls.utilisation)

        # Parse arguments
        while len(argv) > 0:
            switch = argv.pop(0)
            availableSwitches = ["-m", "-i"]

            # Input Validation
            if switch not in availableSwitches:
                raise fftQueryParsingError(cls.utilisation)

            if switch == "-m":

                if len(argv) < 1:
                    raise fftQueryParsingError(cls.utilisation)
                mode_input = argv.pop(0)
                # Check if the mode is valid (and assigns it)
                try:
                    mode = Mode.from_value(str(mode_input))
                except ValueError:
                    raise fftQueryParsingError(cls.utilisation)

            elif switch == "-i":

                if len(argv) < 1:
                    raise fftQueryParsingError(cls.utilisation)

                user_input_filename = str(argv.pop(0))
                image_name = "../Figures/" + user_input_filename
                # Check if the image file exists
                if not os.path.isfile(image_name):
                    raise fftQueryParsingError(
                        f"The specified image file '{user_input_filename}' does not exist."
                    )

                image = cls.convert_image_to_numpy_array(image_name)
            else:
                raise fftQueryParsingError(cls.utilisation)

        return fftQuery(mode, image_name, image)

    @classmethod
    def convert_image_to_numpy_array(cls, image_name: str) -> np.ndarray:
        return np.array(cv2.imread(image_name, cv2.IMREAD_GRAYSCALE))

    def get_mode(self) -> str:
        return self.mode

    def get_image_name(self) -> str:
        return self.image_name

    def get_image(self) -> np.ndarray:
        return self.image


class fftQueryParsingError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
