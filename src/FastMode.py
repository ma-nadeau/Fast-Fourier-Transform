from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt
import os
from FourierTransform import rescale_image_power2, transform_2D


def plot_fast_mode(
    original_image: np.ndarray, image_name: str, save_plot: bool = False
) -> None:
    """Plot the original image and its Fourier transform."""

    # First scale the image
    scaled_image = rescale_image_power2(original_image)

    plt.figure(figsize=(10, 6))
    plt.suptitle("Fast Fourier Transform Analysis", fontsize=18, fontweight="bold")

    # Plot original image
    plt.subplot(1, 2, 1)
    plt.imshow(scaled_image, cmap="gray")
    plt.title("Scaled Original Image", fontsize=14, fontweight="bold")
    plt.axis("off")

    # Plot Custom FFT
    plt.subplot(1, 2, 2)
    plt.imshow(np.abs(transform_2D(scaled_image)), cmap="gray", norm=LogNorm())
    plt.title("FFT (Log Scale)", fontsize=14, fontweight="bold")
    plt.axis("off")

    plt.tight_layout()

    if save_plot:
        folder_path = "../Results"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        plt.savefig(os.path.join(folder_path, f"{image_name}_FastMode_FFT.png"))

    plt.show()

def plot_testing_fast_mode(
    original_image: np.ndarray, image_name: str, save_plot: bool = False
) -> None:
    """Plot the original image and its Fourier transform."""

    # First scale the image
    scaled_image = rescale_image_power2(original_image)

    plt.figure(figsize=(10, 6))
    plt.suptitle("Fast Fourier Transform Analysis", fontsize=18, fontweight="bold")

    # Plot original image
    plt.subplot(1, 3, 1)
    plt.imshow(scaled_image, cmap="gray")
    plt.title("Scaled Original Image", fontsize=14, fontweight="bold")
    plt.axis("off")

    # Plot Custom FFT
    plt.subplot(1, 3, 2)
    plt.imshow(np.abs(transform_2D(scaled_image)), cmap="gray", norm=LogNorm())
    plt.title("FFT (Log Scale)", fontsize=14, fontweight="bold")
    plt.axis("off")

    # Plot Numpy FFT
    plt.subplot(1, 3, 3)
    plt.imshow(np.abs(np.fft.fft2(scaled_image)), cmap="gray", norm=LogNorm())
    plt.title("Numpy FFT (Log Scale)", fontsize=14, fontweight="bold")
    plt.axis("off")

    plt.tight_layout()

    if save_plot:
        folder_path = "../Results"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        plt.savefig(os.path.join(folder_path, f"{image_name}_Testing_FastMode_FFT.png"))

    plt.show()
