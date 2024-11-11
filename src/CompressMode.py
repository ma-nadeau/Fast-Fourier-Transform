import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple
from FastFourierTransform import FastFourierTransform
from DiscreteFourierTransform2D import DiscreteFourierTransform2D

class CompressMode:
    def __init__(
        self,
        original_image: np.ndarray,
        folder_path: str = "../Results",
    ):
        self.original_image = original_image
        self.folder_path = folder_path
        self.fft_image = self.run_FFT()

        self.run_compression()
        pass

    def run_FFT(self) -> np.ndarray:
        fft = FastFourierTransform()
        return fft.fft_2D(self.original_image)
    

    def compression_fft_largest_percentile(
        self, fft_image: np.array, compression_level: float
    ) -> Tuple[np.array, int]:

        flatten_fft = np.abs(fft_image).flatten()

        # Get the threshold value based on the compression level (i.e., the largest percentile)
        threshold = np.percentile(flatten_fft, 100 - compression_level)

        compressed_fft = np.where(np.abs(fft_image) >= threshold, fft_image, 0)

        return compressed_fft, np.count_nonzero(compressed_fft)

    def compression_fft_smallest_percentile_and_portion_largest(
        self, fft_image: np.array, compression_level: float
    ) -> Tuple[np.array, int]:
        # TODO: implement this method
        pass

    def compress_and_inverse_fft(
        self,
        fft_image: np.array,
        compression_level: float,
        use_largest_percentile: bool = True,
    ) -> Tuple[np.array, int]:
        fft = FastFourierTransform()
        dft = DiscreteFourierTransform2D()
        if use_largest_percentile:
            compressed_fft, number_non_zeros = self.compression_fft_largest_percentile(
                fft_image, compression_level
            )
            inverse_fft_image = dft.idft_2D(compressed_fft)
        else:
            # TODO: implement this method
            compressed_fft, number_non_zeros = (
                self.compression_fft_smallest_percentile_and_portion_largest(
                    fft_image, compression_level
                )
            )
        return inverse_fft_image, number_non_zeros

    def plot_images(self) -> None:
        """Plot the original image and its compressed images at different levels."""
        compression_levels = [0, 1, 10, 30, 50, 99.9]
        plt.figure(figsize=(15, 10))
        plt.suptitle("Compressed Image", fontsize=20)

        for i, level in enumerate(compression_levels):
            compressed_image, number_non_zeros = self.compress_and_inverse_fft(
                self.fft_image, level
            )
            print(f"Compression Level: {level}%, Non-Zero Coefficients: {number_non_zeros}")

            plt.subplot(2, 3, i + 1)
            plt.imshow(compressed_image.real, cmap="gray")
            plt.title(f"Compression: {level}%")
            plt.axis("off")

        plt.tight_layout()

        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        plt.savefig(os.path.join(self.folder_path, "Compressed_Images.png"))

    
    def run_compression(self) -> None:
        self.run_FFT()
        self.plot_images()
