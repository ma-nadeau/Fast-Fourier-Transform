from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt
import DiscreteFourierTransform2D as dft2d  # TODO: This is temporary, will be replaced with the FFT
import os


class FastMode:
    def __init__(
        self,
        original_image: np.ndarray,
        folder_path: str = "../Results",
    ):
        self.original_image = original_image
        self.folder_path = folder_path

        self.numpy_fft_image = np.fft.fft2(self.original_image)
        self.our_fft_image = self.run_FFT()
        self.plot_images()

    def run_FFT(self) -> np.ndarray:
        # TODO: This is temporary, will be replaced with our implementation of the FFT
        return np.fft.fft2(self.original_image)

    def plot_images(self) -> None:
        """Plot the original image and its Fourier transform."""
        plt.figure(figsize=(15, 6))

        # Plot original image
        plt.subplot(1, 3, 1)
        plt.imshow(self.original_image, cmap="gray")
        plt.suptitle("Fast Fourier Transform Analysis", fontsize=20)

        plt.title("Original Image", fontsize=16)
        plt.axis("off")

        # Plot Numpy FFT
        plt.subplot(1, 3, 2)
        plt.imshow(np.log(np.abs(self.numpy_fft_image)), cmap="gray", norm=LogNorm())
        plt.title("Numpy FFT (Log Scale)", fontsize=16)
        plt.axis("off")

        # Plot Custom FFT
        plt.subplot(1, 3, 3)
        plt.imshow(np.log(np.abs(self.our_fft_image)), cmap="gray", norm=LogNorm())
        plt.title("Custom FFT (Log Scale)", fontsize=16)
        plt.axis("off")

        plt.tight_layout()

        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        plt.savefig(os.path.join(self.folder_path, "FastMode_FFT.png"))
