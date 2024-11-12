import numpy as np


class DiscreteFourierTransform:
    def __init__(self):
        pass

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

        # Create vectors for k, l, m, and n
        k = np.arange(M).reshape((M, 1))  # k = [[0], [1], [2], ..., [M-1]]
        m = np.arange(M).reshape((1, M))  # m = [0, 1, 2, ..., M-1]

        # k * m =
        # [[0×0, 0×1, 0×2, ..., 0×(M-1)],
        #  [1×0, 1×1, 1×2, ..., 1×(M-1)],
        #  [2×0, 2×1, 2×2, ..., 2×(M-1)],
        #  ...,
        #  [(M-1)×0, (M-1)×1, (M-1)×2, ..., (M-1)×(M-1)]]
        matrix_km = k * m
        # exp_km = exp(-2i * pi * k * m / M)
        exp_km = np.exp(-2j * np.pi * matrix_km / M)

        l = np.arange(N).reshape((N, 1))  # l = [[0], [1], [2], ..., [N-1]]
        n = np.arange(N).reshape((1, N))  # n = [0, 1, 2, ..., N-1]

        # l is a column vector with shape (N, 1)
        # n is a row vector with shape (1, N)
        # l * n is a matrix with shape (N, N)
        # l * n =
        # [[0×0, 0×1, 0×2, ..., 0×(N-1)],
        #  [1×0, 1×1, 1×2, ..., 1×(N-1)],
        #  [2×0, 2×1, 2×2, ..., 2×(N-1)],
        #  ...,
        #  [(N-1)×0, (N-1)×1, (N-1)×2, ..., (N-1)×(N-1)]]
        matrix_ln = l * n
        # exp_ln = exp(-2i * pi * l * n / N)
        exp_ln = np.exp(-2j * np.pi * matrix_ln / N)

        # Perform the matrix multiplications
        # transformed_signal = sum (sum (exp(-2j * pi * k * m / M) * signal * exp(-2j * pi * l * n / N)))
        transformed_signal = exp_km @ signal @ exp_ln
        return transformed_signal

    def idft_2D(self, transformed_signal : np.ndarray) -> np.ndarray:
        """
        Computes the 2D Inverse Discrete Fourier Transform (IDFT) of the transformed signal.

        This method reconstructs the original time domain signal from its frequency domain representation.

        Returns:
            np.ndarray: The reconstructed 2D signal in the time domain.
        """
        M, N = transformed_signal.shape
        signal = np.zeros((M, N), dtype=complex)

        k = np.arange(M).reshape((M, 1))  # k = [[0], [1], [2], ..., [M-1]]
        m = np.arange(M).reshape((1, M))  # m = [0, 1, 2, ..., M-1]

        # k * m =
        # [[0×0, 0×1, 0×2, ..., 0×(M-1)],
        #  [1×0, 1×1, 1×2, ..., 1×(M-1)],
        #  [2×0, 2×1, 2×2, ..., 2×(M-1)],
        #  ...,
        #  [(M-1)×0, (M-1)×1, (M-1)×2, ..., (M-1)×(M-1)]]
        matrix_km = k * m
        # TODO: Check if it is 12 or 1 * 2
        # exp_kn = exp(2i * pi * k * m / M)
        exp_km = np.exp(2j * np.pi * matrix_km / M)
        
        
        l = np.arange(N).reshape((N, 1))  # l = [[0], [1], [2], ..., [N-1]]
        n = np.arange(N).reshape((1, N))  # n = [0, 1, 2, ..., N-1]
        
        # l is a column vector with shape (N, 1)
        # n is a row vector with shape (1, N)
        # l * n is a matrix with shape (N, N)
        # l * n =
        # [[0×0, 0×1, 0×2, ..., 0×(N-1)],
        #  [1×0, 1×1, 1×2, ..., 1×(N-1)],
        #  [2×0, 2×1, 2×2, ..., 2×(N-1)],
        #  ...,
        #  [(N-1)×0, (N-1)×1, (N-1)×2, ..., (N-1)×(N-1)]]
        matrix_ln = l * n
        # exp_ln = exp(2i * pi * l * n / N)
        exp_ln = np.exp(2j * np.pi * matrix_ln / N)

        # Perform the matrix multiplications
        # f[m,n] = sum (sum F[k,l] * exp(12i * pi * k * m / M) * exp(12i * pi * l * n / N) / (M * N)
        signal = exp_km @ transformed_signal @ exp_ln / (M * N)
        
        return signal
