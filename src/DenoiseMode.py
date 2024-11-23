import numpy as np
import matplotlib.pyplot as plt
import os
from FourierTransform import transform_2D, rescale_image_power2, inverse_transform_2D


def denoise_image(image: np.ndarray) -> np.ndarray:
    """Denoise a given image by setting to zero FFT coefficients
     associated with frequencies out of the [-pi/7,pi/7] range."""
    
    # First switch to frequency domain
    fft = transform_2D(image)

    # Shift the low frequencies to the center
    shifted_fft = np.fft.fftshift(fft)

    # Calculate the indices of the coefficients to keep
    # Here we are deciding to keep only the coefficients with absolute frequency at most pi/7
    pi_percentage = 1/7
    height, width = shifted_fft.shape
    height_offset = round(height//2 * pi_percentage)
    width_offset = round(width//2 * pi_percentage)

    # Determine the rectangle of low frequencies to keep with a 1 mask
    mask = np.zeros(shifted_fft.shape)
    mask[height//2-height_offset:height//2+height_offset, width//2-width_offset:width//2+width_offset] = 1

    # Apply the filtering
    filtered_fft = shifted_fft * mask

    # Shift back to obtain the configuration of frequencies of the original fft
    filtered_fft = np.fft.ifftshift(filtered_fft)
    

    # Determine non_zeroes
    non_zeroes = height_offset * width_offset * 4

    return np.real(inverse_transform_2D(filtered_fft)), non_zeroes


def plot_denoise_mode(
    original_image: np.ndarray, image_name: str, save_plot: bool = False
) -> None:
    """Plot the original image alongside its Denoised version."""

    # First scale the image
    scaled_image = rescale_image_power2(original_image)
    denoised_image, non_zeroes = denoise_image(scaled_image)

    plt.figure(figsize=(10, 6))
    plt.suptitle("Image Denoising Analysis", fontsize=18, fontweight="bold")

    # Plot original image
    plt.subplot(1, 2, 1)
    plt.imshow(scaled_image, cmap="gray")
    plt.title("Scaled Original Image", fontsize=14, fontweight="bold")
    plt.axis("off")

    # Plot Denoised image
    plt.subplot(1, 2, 2)
    plt.imshow(denoised_image, cmap="gray")
    plt.title("Denoised Image", fontsize=14, fontweight="bold")
    plt.axis("off")

    # Print the number of non-zeroes used and their percentage
    print(f"DENOISE MODE: The number of non-zeroes used: {non_zeroes}.")
    print(
        f"They represent: {(non_zeroes/(scaled_image.shape[0] * scaled_image.shape[1]) * 100):.2f}% of the original Fourrier coefficients."
    )

    plt.tight_layout()
    if save_plot:
        folder_path = "../Results"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        plt.savefig(os.path.join(folder_path, f"{image_name}_Denoised_Image.png"))
    plt.show()
