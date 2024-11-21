import numpy as np
import matplotlib.pyplot as plt
from PlottingMode import create_2D_array_of_random_element
from FourierTransform import transform_2D
import os

def compute_error(benchmark_model: np.ndarray, custom_model: np.ndarray) -> np.ndarray:
    """Compute the error between the FFT and Naive Fourier Transform."""
    return np.abs(benchmark_model - custom_model) / np.abs(custom_model)


def plot_error(
    fft_model: np.ndarray, custom_fft_model: np.ndarray, custom_dft_model: np.ndarray
) -> None:
    """Plot the error between the FFT, custom FFT, and custom DFT."""
    error_fft = compute_error(fft_model, custom_fft_model)
    error_dft = compute_error(fft_model, custom_dft_model)

    _, axs = plt.subplots(1, 2, figsize=(12, 6))

    image0 = axs[0].imshow(error_fft, cmap="viridis")
    axs[0].set_title("Error Custom FFT", fontsize=14, fontweight="bold")
    axs[0].set_xlabel("X-axis")
    axs[0].set_ylabel("Y-axis")
    plt.colorbar(image0, ax=axs[0])

    image1 = axs[1].imshow(error_dft, cmap="viridis")
    axs[1].set_title("Error Custom DFT", fontsize=14, fontweight="bold")
    axs[1].set_xlabel("X-axis")
    axs[1].set_ylabel("Y-axis")
    plt.colorbar(image1, ax=axs[1])

    plt.suptitle(
        "Error Comparison of custom model with numpy FFT",
        fontsize=18,
        fontweight="bold",
    )
    
    folder_path = "../Results"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.savefig(os.path.join(folder_path, "Error-Comparison-of-Custom-Models-with-NumPy-FFT.png"))
    plt.show()
    


if __name__ == "__main__":
    # Create a 2D array of random elements
    array = create_2D_array_of_random_element(5)

    # Compute the Fast Fourier Transform using NumPy
    fft_model = np.fft.fft2(array)

    # Compute the Naive Fourier Transform
    fast_model = transform_2D(array, True)  # Fast Fourier Transform
    naive_model = transform_2D(array, False)  # Naive Fourier Transform

    # Compute the error between the FFT and Naive Fourier Transform
    plot_error(fft_model, fast_model, naive_model)
