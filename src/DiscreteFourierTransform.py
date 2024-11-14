import numpy as np


class DiscreteFourierTransform:
    def __init__(self):
        pass

    def dft_1D(self, signal: np.ndarray) -> np.ndarray:
        """Computes the 1D Discrete Fourier Transform (DFT) of the input signal.

        This method transforms the spatial domain signal into its frequency domain representation.

        Returns:
            np.ndarray: The transformed 1D signal in the frequency domain.
        """

        N = signal.size

        transformed_signal = np.zeros(N, dtype=complex)

        # Create a vector for k such that k = 0, 1, 2, ..., N-1
        k = np.arange(N)  # k = [0, 1, 2, ..., N-1]
        n = np.arange(N)  # n = [0, 1, 2, ..., N-1]

        # k outer n =
        # [[0×0, 0×1, 0×2, ..., 0×(N-1)],
        #  [1×0, 1×1, 1×2, ..., 1×(N-1)],
        #  [2×0, 2×1, 2×2, ..., 2×(N-1)],
        #  ...,
        #  [(N-1)×0, (N-1)×1, (N-1)×2, ..., (N-1)×(N-1)]]
        matrix_kn = np.outer(k, n)

        # exp_kn = exp(-2i * pi * k * n / N)
        exp_kn = np.exp(-2j * np.pi * matrix_kn / N)

        transformed_signal = exp_kn @ signal
        return transformed_signal

    def idft_1D(self, transformed_signal: np.ndarray) -> np.ndarray:
        """
        Computes the 1D Inverse Discrete Fourier Transform (IDFT) of the transformed signal.

        This method reconstructs the original time domain signal from its frequency domain representation.

        Returns:
            np.ndarray: The reconstructed 1D signal in the time domain.
        """

        N = transformed_signal.size
        signal = np.zeros(N, dtype=complex)

        k = np.arange(N)  # k = [0, 1, 2, ..., N-1]
        n = np.arange(N)  # n = [0, 1, 2, ..., N-1]

        # k outer n =
        # [[0×0, 0×1, 0×2, ..., 0×(N-1)],
        #  [1×0, 1×1, 1×2, ..., 1×(N-1)],
        #  [2×0, 2×1, 2×2, ..., 2×(N-1)],
        #  ...,
        #  [(N-1)×0, (N-1)×1, (N-1)×2, ..., (N-1)×(N-1)]]

        matrix_kn = np.outer(k, n)

        # exp_kn = exp(2i * pi * k * n / N) for k, n = 0, 1, 2, ..., N-1
        exp_kn = np.exp(2j * np.pi * matrix_kn / N)
        signal = exp_kn @ transformed_signal / N
        return signal

    def dft_2D(self, signal: np.ndarray) -> np.ndarray:
        """
        Computes the 2D Discrete Fourier Transform (DFT) of the input signal.

        This method transforms the spatial domain signal into its frequency domain representation.

        Returns:
            np.ndarray: The transformed 2D signal in the frequency domain.
        """
        M, N = signal.shape
        # Initialize the transformed signal
        transformed_signal = np.zeros((M, N), dtype=complex)
        
        # axis = 1 means apply the function along the rows
        # Apply the 1D DFT to each row of the signal
        intermediate = np.apply_along_axis(self.dft_1D, axis=1, arr=signal)

        # axis = 0 means apply the function along the columns
        # Apply the 1D DFT to each column of the intermediate result
        transformed_signal = np.apply_along_axis(self.dft_1D, axis=0, arr=intermediate)
        
        return transformed_signal

    def idft_2D(self, transformed_signal: np.ndarray) -> np.ndarray:
        """
        Computes the 2D Inverse Discrete Fourier Transform (IDFT) of the transformed signal.

        This method reconstructs the original time domain signal from its frequency domain representation.

        Returns:
            np.ndarray: The reconstructed 2D signal in the time domain.
        """
        M, N = transformed_signal.shape
        signal = np.zeros((M, N), dtype=complex)
        
        # axis = 1 means apply the function along the rows
        intermediate = np.apply_along_axis(self.idft_1, axis=1, arr=transformed_signal)
        # axis = 0 means apply the function along the columns
        signal = np.apply_along_axis(self.idft_1D, axis=0, arr=intermediate)
        return signal
