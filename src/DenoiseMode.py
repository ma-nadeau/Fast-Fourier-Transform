from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt
import DiscreteFourierTransform2D as dft2d # This is temporary, will be replaced with the FFT
import os


class DenoiseMode:
    def __init__(
        self,
        original_image: np.ndarray,
        folder_path: str = "../Results",
    ):
        self.original_image = original_image
        self.folder_path = folder_path
        self.denoise_image = self.run_denoise()

        self.plot_images()
        pass

    def run_denoise(self):
        #TODO: This is temporary, will be replaced with our FFT
    
        transformed_signal = np.fft.fft2(self.original_image)
        
        # Apply a simple denoising technique (e.g., thresholding)
        threshold = np.mean(transformed_signal) + 2 * np.std(transformed_signal) # TODO: check what high frequencies are
        denoised_signal = np.where(transformed_signal > threshold, transformed_signal, 0)
        
        #TODO: This is temporary, will be replaced with our FFT
        # Inverse DFT to get the denoised image
        denoised_image = np.fft.ifft2(denoised_signal).real
        
        return denoised_image

    def plot_images(self):
        """Plot the original image and its Denoise image."""
        denoised_image = self.run_denoise()
        plt.figure(figsize=(10, 5))

        # Plot original image
        plt.subplot(1, 2, 1)
        plt.imshow(self.original_image, cmap="gray")
        plt.title("Original Image")
        plt.axis("off")

        # Plot denoised image
        plt.subplot(1, 2, 2)
        plt.imshow(denoised_image, cmap="gray")
        plt.title("Denoised Image")
        plt.axis("off")

        plt.tight_layout()
        
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
        plt.savefig(os.path.join(self.folder_path, "Denoised_Image.png"))
        
        plt.show()
