import numpy as np
import matplotlib.pyplot as plt
import os


class DenoiseMode:
    def __init__(
        self,
        original_image: np.ndarray,
        folder_path: str = "../Results",
    ):
        self.original_image = original_image
        self.folder_path = folder_path
        self.denoised_image = self.denoise()

        self.run_denoise()

    def denoise(self) -> np.ndarray:

        # FIXME: This is temporary, will be replaced with our FFT
        transformed_signal = np.fft.fft2(self.original_image)

        # Apply a simple denoising technique (e.g., thresholding)
        # TODO: What is considered a high frequency?
        threshold = np.mean(transformed_signal) + 2 * np.std(transformed_signal)
        denoised_signal = np.where(
            transformed_signal > threshold, transformed_signal, 0
        )

        # FIXME: This is temporary, will be replaced with our FFT
        # Inverse DFT to get the denoised image
        denoised_image = np.fft.ifft2(denoised_signal).real

        return denoised_image

    def plot_images(self) -> None:
        """Plot the original image and its Denoise image."""
        plt.figure(figsize=(10, 5))

        # Plot original image
        plt.suptitle("Denoised Image", fontsize=20)

        # Plot original image
        plt.subplot(1, 2, 1)
        plt.imshow(self.original_image, cmap="gray")
        plt.title("Original Image")
        plt.axis("off")

        # Plot denoised image
        plt.subplot(1, 2, 2)
        plt.imshow(self.denoised_image, cmap="gray")
        plt.title("Denoised Image")
        plt.axis("off")

        plt.tight_layout()

        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        plt.savefig(os.path.join(self.folder_path, "Denoised_Image.png"))

    def run_denoise(self) -> None:
        self.denoise()
        self.plot_images()
