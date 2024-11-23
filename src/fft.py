""" Python program that implemenets the Fast Fourier Transform (FFT) algorithm. """

import sys
from CompressMode import plot_compression_mode
from DenoiseMode import plot_denoise_mode
from PlottingMode import PlottingMode
from fftQuery import fftQuery, fftQueryParsingError
from FastMode import plot_fast_mode
from QueryMode import Mode


def main() -> None:
    """ Main logic of the fft program """
    try:
        query = fftQuery.parseArguments(sys.argv[1:])
    except fftQueryParsingError as error:
        print("ERROR\tIncorrect input syntax: " + str(error))
        exit()
    
    # Fetch image
    image_name = query.image_name
    original_image = fftQuery.convert_image_to_numpy_array(image_name)
    
    # Get the image name without the path and extension
    image_name = image_name.split("/")[-1].split(".")[0]

    if query.mode == Mode.FAST:
        plot_fast_mode(original_image, image_name)
    elif query.mode == Mode.DENOISE:
        plot_denoise_mode(original_image, image_name)
    elif query.mode == Mode.COMPRESS:
        plot_compression_mode(original_image, image_name)
    elif query.mode == Mode.PLOT_RUNTIME:
        plotter = PlottingMode()
        plotter.plot_average_runtime()
    else:
        print("ERROR\tUnknown mode: " + query.mode)
        exit()


# For when the program is invoked from the command line (stdin)
if __name__ == "__main__":
    main()
