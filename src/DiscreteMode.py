from DiscreteFourierTransform import DiscreteFourierTransform
from FastFourierTransform import FastFourierTransform
import numpy as np
from typing import Tuple


class DiscreteMode:

    def __init__(self):
        X = self.create_random_signal(2**3)  # Create a random signal of size 8
        error = self.compare_dft_fft(X)  # Compare DFT and FFT
        print(f"Error between DFT and FFT: {error}")

    def compare_dft_fft(self, X: np.ndarray) -> float:
        """Compare the Discrete Fourier Transform (DFT) and Fast Fourier Transform (FFT) of a 1D signal."""
        dft = DiscreteFourierTransform()
        fft = FastFourierTransform()

        # Compute the DFT of the signal
        X_dft = dft.dft_1D(X)
        print(f"DFT result: {X_dft}")

        # Compute the FFT of the signal
        X_fft = fft.fft_1D_matrix(X)
        print(f"FFT result: {X_fft}")

        # Compute the error between DFT and FFT results
        error = np.linalg.norm(X_dft - X_fft)
        return error

    def compare_idft_ifft(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compare the Inverse Discrete Fourier Transform (IDFT) and Inverse Fast Fourier Transform (IFFT) of a 1D signal."""
        dft = DiscreteFourierTransform()
        fft = FastFourierTransform()

        # Compute the IDFT of the signal
        X_idft = dft.idft_1D(X)

        # Compute the IFFT of the signal
        X_ifft = fft.ifft_1D(X)

        return X_idft, X_ifft

    def create_random_signal(self, N: int) -> np.ndarray:
        """Create a random 1D signal of size N."""
        return np.random.rand(N) + 1j * np.random.rand(N)

    def error_dft_fft(self, X: np.ndarray) -> float:
        """Compute the error between the Discrete Fourier Transform (DFT) and Fast Fourier Transform (FFT) of a 1D signal."""
        X_dft, X_fft = self.compare_dft_fft(X)
        error = np.linalg.norm(X_dft - X_fft)
        return error
