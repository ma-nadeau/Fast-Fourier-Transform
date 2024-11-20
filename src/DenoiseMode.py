import numpy as np
import matplotlib.pyplot as plt
import os
from FastFourierTransform import FastFourierTransform
from FourierTransform import transform_2D, rescale_image_power2, inverse_transform_2D

def denoise_image(image : np.ndarray):
    # First switch to frequency domain
    fft = transform_2D(image)

    # Get rid of high frequencies' contributions
    # Here we are considering that the frequencies above 15/16pi are high frequencies
    pi_percentage = 1/6
    threshold_lo = pi_percentage * np.pi
    threshold_hi = (2 - pi_percentage) * np.pi
    non_zeroes = fft.shape[0] * fft.shape[1] # To determine the number of non-zeros being used
    for row in range(fft.shape[0]):
        for col in range(fft.shape[1]):
            if threshold_lo < row*2*np.pi/fft.shape[0] < threshold_hi\
                or threshold_lo < col*2*np.pi/fft.shape[1] < threshold_hi:
                fft[row, col] = 0
                non_zeroes -= 1

    return np.real(inverse_transform_2D(fft)), non_zeroes

def plot_denoise_mode(original_image):
    """Plot the original image and its Denoise image."""

    # First scale the image
    scaled_image = rescale_image_power2(original_image)
    denoised_image, non_zeroes = denoise_image(scaled_image)

    plt.figure(figsize=(10,6))
    plt.suptitle("Image Denoising Analysis", fontsize=18)

    # Plot original image
    plt.subplot(1, 2, 1)
    plt.imshow(scaled_image, cmap="gray")
    plt.title("Scaled Original Image", fontsize=14)
    plt.axis("off")
    
    # Plot Custom FFT
    plt.subplot(1, 2, 2)
    plt.imshow(denoised_image, cmap="gray")
    plt.title("Denoised Image", fontsize=14)
    plt.axis("off")
    
    # Print the number of non-zeroes used and their percentage
    print(f"DENOISE MODE: The number of non-zeroes used: {non_zeroes}.")
    print(f"They represent: {(non_zeroes/(scaled_image.shape[0] * scaled_image.shape[1]) * 100):.2f}% of the original Fourrier coefficients.")

    plt.tight_layout()

    plt.show()
