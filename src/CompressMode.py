import numpy as np
import matplotlib.pyplot as plt
from FourierTransform import inverse_transform_2D, rescale_image_power2, transform_2D

def compress_image(image : np.ndarray, percentile_percentage : int):
    # First switch to frequency domain
    fft = transform_2D(image)
    magnitude_fft = np.abs(fft)
    
    # Find the threshold value for the given percentile
    threshold = np.percentile(magnitude_fft, percentile_percentage)

    # Set elements below the threshold to 0
    magnitude_fft = np.where(magnitude_fft < threshold, 0, magnitude_fft)
    fft = np.where(magnitude_fft == 0, 0, fft)

    # Keep track of the number of values not set to 0
    non_zero_count = np.count_nonzero(fft)
    return np.real(inverse_transform_2D(fft)), non_zero_count

def plot_compression_mode(original_image : np.ndarray):
    """Plot the original image and its compressed images at different levels."""
    # First scale the image
    scaled_image = rescale_image_power2(original_image)
    
    compression_levels = [0, 40, 75, 90, 99, 99.9]

    plt.figure(figsize=(15, 10))
    plt.suptitle("Compressed Images", fontsize=18)
    for i, level in enumerate(compression_levels):
        compressed_image, non_zero_count = compress_image(scaled_image, level)

        # Plot compressed image
        plt.subplot(2, 3, 1 + i)
        plt.imshow(compressed_image, cmap="gray")
        plt.title(f"Compression: {level}%")
        plt.axis("off")
        
        print(f"COMPRESSION MODE: The number of non-zeroes used: {non_zero_count}.")
        print(f"They represent: {(non_zero_count/(scaled_image.shape[0] * scaled_image.shape[1]) * 100):.2f}% of the original Fourier coefficients.")
    plt.tight_layout()
    plt.show()
