from typing import List
import os
import cv2
import numpy as np
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

    def __init__(
        self, mode: Mode = Mode.FAST, image_name: str = "../Figures/moonlanding.png"
    ):
        self.mode = mode
        self.image_name = image_name

    @classmethod
    def parseArguments(cls, argv: List[str]) -> "fftQuery":
        # Check that the argument length is valid
        if not 0 <= len(argv) <= 4:
            raise fftQueryParsingError(cls.utilisation)
        optionalArgs = cls.parseOptionalArguments(argv)

        return fftQuery(**optionalArgs)

    @classmethod
    def parseOptionalArguments(cls, argv: list[str]) -> dict:
        optionalArgs = {}

        while len(argv) > 0:
            switch = argv.pop(0)
            availableSwitches = ["-m", "-i"]
            # Input validation
            if switch not in availableSwitches or len(argv) < 1:
                raise fftQueryParsingError(cls.utilisation)

            match (switch):
                case "-m":
                    if optionalArgs.get("mode"):
                        raise fftQueryParsingError(cls.utilisation)
                    try:
                        optionalArgs["mode"] = Mode.from_value(int(argv.pop(0)))
                    except ValueError as e:
                        raise fftQueryParsingError(e)
                case "-i":
                    if optionalArgs.get("image_name"):
                        raise fftQueryParsingError(cls.utilisation)
                    image_name = "../Figures/" + argv.pop(0)
                    
                    # Check if the image file exists
                    if not os.path.isfile(image_name):
                        raise fftQueryParsingError(
                            f"The specified image file '{image_name}' does not exist."
                        )
                    optionalArgs["image_name"] = image_name
        return optionalArgs

    @staticmethod
    def convert_image_to_numpy_array(image_name: str) -> np.ndarray:
        return np.array(cv2.imread(image_name, cv2.IMREAD_GRAYSCALE))

    def __repr__(self):
        return f"[mode:{self.mode.name}, image:{self.image_name}]"


class fftQueryParsingError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
