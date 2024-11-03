from typing import NamedTuple, List, Dict
import os


class fftQuery:
    # Program utilization reminder
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

    def __init__(self, mode: str = "1", image: str = "moonlanding.png"):
        self.mode = mode
        self.image = image

    @classmethod
    def parseArguments(cls, argv: List[str]):
        mode = "1"
        image = "moonlanding.png"

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
                mode = argv.pop(0)
                if mode not in ["1", "2", "3", "4"]:
                    raise fftQueryParsingError(cls.utilisation)
            elif switch == "-i":
                if len(argv) < 1:
                    raise fftQueryParsingError(cls.utilisation)
                image = argv.pop(0)

                # Check if the image file exists
                if not os.path.isfile(image):
                    raise fftQueryParsingError(
                        f"The specified image file '{image}' does not exist."
                    )
            else:
                raise fftQueryParsingError(cls.utilisation)

        return fftQuery(mode, image)

    def get_mode(self) -> str:
        return self.mode

    def get_image(self) -> str:
        return self.image


class fftQueryParsingError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
