from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt
import DiscreteFourierTransform2D as dft2d # This is temporary, will be replaced with the FFT
import os


class FastMode:
    def __init__(
        self,
        original_image: np.ndarray,
        folder_path: str = "../Results",
    ):
        self.original_image = original_image
        self.folder_path = folder_path
        self.fft_image = self.run_FFT()

        self.plot_images()
        pass

    def run_FFT(self):
        
        # This is temporary, will be replaced with the FFT
        DiscreteFourierTransform2D = dft2d.DiscreteFourierTransform2D(
            self.original_image
        )
        return DiscreteFourierTransform2D.compute_dft_2d().get_transformed_signal()

    def plot_images(self):
        """Plot the original image and its Fourier transform."""
        plt.figure(figsize=(10, 5))

        # Plot original image
        plt.subplot(1, 2, 1)
        plt.imshow(self.original_image, cmap="gray")
        plt.title("Original Image")
        plt.axis("off")

        # Plot Fourier Transform
        plt.subplot(1, 2, 2)
        plt.imshow(self.fft_image, cmap="gray", norm=LogNorm())
        plt.title("Fourier Transform (Log Scale)")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        plt.savefig(os.path.join(self.folder_path, "FastMode_FFT.png"))
