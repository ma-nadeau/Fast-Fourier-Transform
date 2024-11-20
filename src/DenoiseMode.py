import numpy as np
import matplotlib.pyplot as plt
import os
from FastFourierTransform import FastFourierTransform
from FourierTransform import transform_2D, rescale_image_power2, inverse_transform_2D


    # fft = FastFourierTransform()
    # dft = DiscreteFourierTransform()
    # transformed_signal = fft.fft_2D(self.original_image)
    # # TODO: What is considered a high frequency?
    # threshold = np.mean(transformed_signal) + 2 * np.std(transformed_signal)
    # denoised_signal = np.where(
    #     transformed_signal > threshold, transformed_signal, 0
    # )
    # denoised_image = dft.idft_2D(denoised_signal)
    # return denoised_image.real

def denoise_image(image : np.ndarray):
    # First switch to frequency domain
    fft = transform_2D(image)

    # Get rid of high frequencies' contributions
    # Here we are considering that the frequencies above 15/16pi are high frequencies
    pi_percentage = 1/4
    threshold_lo = pi_percentage * np.pi
    threshold_hi = (2 - pi_percentage) * np.pi
    counter = fft.shape[0] * fft.shape[1] # To determine the number of non-zeros being used
    for row in range(fft.shape[0]):
        for col in range(fft.shape[1]):
            if threshold_lo < row*2*np.pi/fft.shape[0] < threshold_hi\
                or threshold_lo < col*2*np.pi/fft.shape[1] < threshold_hi:
                fft[row, col] = 0
                counter -= 1
    
    # Print the number of non-zeroes used and their percentage
    print(f"DENOISE MODE: The number of non-zeroes used: {counter}.")
    print(f"They represent: {(counter/(fft.shape[0] * fft.shape[1]) * 100):.2f}% of the original Fourier coefficients.")

    return inverse_transform_2D(fft)

def plot_denoise_mode(original_image):
    """Plot the original image and its Denoise image."""

    # First scale the image
    scaled_image = rescale_image_power2(original_image)
    denoised_image = denoise_image(scaled_image)

    plt.figure(figsize=(10,6))
    plt.suptitle("Image Denoising Analysis", fontsize=18)

    # Plot original image
    plt.subplot(1, 2, 1)
    plt.imshow(scaled_image, cmap="gray")
    plt.title("Scaled Original Image", fontsize=14)
    plt.axis("off")
    
    # Plot Custom FFT
    plt.subplot(1, 2, 2)
    plt.imshow(np.abs(denoised_image), cmap="gray")
    plt.title("Denoised Image", fontsize=14)
    plt.axis("off")

    plt.tight_layout()

    plt.show()
