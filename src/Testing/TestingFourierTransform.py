import os
import sys

script_dir = os.path.dirname(__file__)
src_dir = os.path.join(script_dir, '..')
sys.path.append(src_dir)

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from DenoiseMode import denoise_image
from fftQuery import fftQuery
from PlottingMode import create_2D_array_of_random_element
from FourierTransform import rescale_image_power2, transform_2D, inverse_transform_2D


def compute_difference(
    benchmark_model: np.ndarray, custom_model: np.ndarray
) -> np.ndarray:
    """Compute the error between the FFT and Naive Fourier Transform."""
    return np.abs(benchmark_model - custom_model)


def plot_error(
    fft_model: np.ndarray,
    custom_fft_model: np.ndarray,
    custom_dft_model: np.ndarray,
    isInverse: bool = False,
) -> None:
    """Plot the error between the FFT, custom FFT, and custom DFT."""
    error_fft = compute_difference(fft_model, custom_fft_model)
    error_dft = compute_difference(fft_model, custom_dft_model)

    _, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Set the title and subtitle based on the type of transform
    if isInverse:
        subtitleDFT = "Error Custom IDFT"
        subtitleFFT = "Error Custom IFFT"
        filename = "Error-Comparison-of-Custom-Models-with-NumPy-Inverse.png"
        title = (
            "Error Comparision of Custom Models with NumPy Inverse Fourier Transform"
        )
    else:
        subtitleDFT = "Error Custom DFT"
        subtitleFFT = "Error Custom FFT"
        filename = "Error-Comparison-of-Custom-Models-with-NumPy.png"
        title = "Error Comparision of Custom Models with NumPy Fourier Transform"

    # Plot the error between the FFT and Naive Fourier Transform
    image0 = axs[0].imshow(error_fft, cmap="viridis")
    axs[0].set_title(subtitleFFT, fontsize=14, fontweight="bold")
    axs[0].set_xlabel("X-axis")
    axs[0].set_ylabel("Y-axis")
    plt.colorbar(image0, ax=axs[0])

    # Plot the error between the DFT and Naive Fourier Transform
    image1 = axs[1].imshow(error_dft, cmap="viridis")
    axs[1].set_title(subtitleDFT, fontsize=14, fontweight="bold")
    axs[1].set_xlabel("X-axis")
    axs[1].set_ylabel("Y-axis")
    plt.colorbar(image1, ax=axs[1])

    plt.suptitle(
        title,
        fontsize=18,
        fontweight="bold",
    )

    folder_path = "../Results"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.savefig(os.path.join(folder_path, filename))

def plot_denoising_test(original_image : np.ndarray):

    # First scale the image
    scaled_image = rescale_image_power2(original_image)
    
    # Apply Gaussian filter for denoising
    sigma = 2.3  # Standard deviation for Gaussian kernel
    scipy_denoised_image = gaussian_filter(scaled_image, sigma=sigma)
    denoised_image = denoise_image(scaled_image)[0]

    plt.figure(figsize=(10, 6))
    plt.suptitle("Image Denoising Testing", fontsize=18, fontweight="bold")

    # Plot original image
    plt.subplot(1, 3, 1)
    plt.imshow(scaled_image, cmap="gray")
    plt.title("Scaled Original Image", fontsize=14, fontweight="bold")
    plt.axis("off")

    # Plot Scipy Denoised image
    plt.subplot(1, 3, 2)
    plt.imshow(scipy_denoised_image, cmap="gray")
    plt.title("Scipy Denoised Image", fontsize=14, fontweight="bold")
    plt.axis("off")

    # Plot Denoised image
    plt.subplot(1, 3, 3)
    plt.imshow(denoised_image, cmap="gray")
    plt.title("Custom Denoised Image", fontsize=14, fontweight="bold")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()


utilisation = """ The input should have the following format:
    python TestingFourierTransform.py <test_num>
    
    <test_num> (mandatory):
    - 1 Plot tests for 2D Fourier Transforms and inverses.
    - 2 Plot DenoiseMode testing.
    """

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(utilisation)
        exit()

    test_num = sys.argv[1]

    if test_num == '1':
        # Create a 2D array of random elements
        array = create_2D_array_of_random_element(5)

        # Compute the Fast Fourier Transform using NumPy
        fft_model = np.fft.fft2(array)

        # Compute the Naive Fourier Transform
        fast_model = transform_2D(array, True)  # Fast Fourier Transform
        naive_model = transform_2D(array, False)  # Naive Fourier Transform
        # Compute the error between the FFT and Naive Fourier Transform
        plot_error(fft_model, fast_model, naive_model)

        # Compute the Fast Fourier Transform using NumPy
        inverse_fft_model = np.fft.ifft2(fft_model)

        # Compute the Naive Fourier Transform
        inverse_fast_model = inverse_transform_2D(
            fast_model, True
        )  # Fast Fourier Transform
        inverse_naive_model = inverse_transform_2D(
            naive_model, False
        )  # Naive Fourier Transform

        plot_error(inverse_fft_model, inverse_fast_model, inverse_naive_model, True)
    elif test_num == '2':
        plot_denoising_test(fftQuery.convert_image_to_numpy_array(f'{os.path.dirname(__file__)}\\..\\..\\Figures\\moonlanding.png'))
