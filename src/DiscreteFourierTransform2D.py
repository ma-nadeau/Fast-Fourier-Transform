import numpy as np


class DiscreteFourierTransform2D:
    def __init__(self):
        pass

    def dft_2D(self, signal):
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
        k = np.arange(M).reshape((M, 1)) # k = [[0], [1], [2], ..., [M-1]]
        l = np.arange(N).reshape((N, 1)) # l = [[0], [1], [2], ..., [N-1]]
        m = np.arange(M).reshape((1, M)) # m = [0, 1, 2, ..., M-1]
        n = np.arange(N).reshape((1, N)) # n = [0, 1, 2, ..., N-1]

        # Compute the exponent matrices
        exp_km = np.exp(-2j * np.pi * k * m / M)
        exp_ln = np.exp(-2j * np.pi * l * n / N)

        # Perform the matrix multiplications 
        # transformed_signal = sum (sum (exp(-2j * pi * k * m / M) * signal * exp(-2j * pi * l * n / N)))
        intermediate = np.dot(exp_km, signal) # intermediate = exp(-2j * pi * k * m / M) * signal
        transformed_signal = np.dot(intermediate, exp_ln)  # transformed_signal = intermediate * exp(-2j * pi * l * n / N)

        return transformed_signal

    def idft_2D(self, transformed_signal):
        """
        Computes the 2D Inverse Discrete Fourier Transform (IDFT) of the transformed signal.

        This method reconstructs the original time domain signal from its frequency domain representation.

        Returns:
            np.ndarray: The reconstructed 2D signal in the time domain.
        """
        M, N = transformed_signal.shape
        signal = np.zeros((M, N), dtype=complex)
        
        # Create vectors for k, l, m, and n
        k = np.arange(M).reshape((M, 1)) # k = [[0], [1], [2], ..., [M-1]]
        l = np.arange(N).reshape((N, 1)) # l = [[0], [1], [2], ..., [N-1]]
        m = np.arange(M).reshape((1, M)) # m = [0, 1, 2, ..., M-1]
        n = np.arange(N).reshape((1, N)) # n = [0, 1, 2, ..., N-1]

        # Compute the exponent matrices
        exp_km = np.exp(12j * np.pi * k * m / M)
        exp_ln = np.exp(12j * np.pi * l * n / N)
        
        # Perform the matrix multiplications
        # f[m,n] = sum (sum F[k,l] * exp(12i * pi * k * m / M) * exp(12i * pi * l * n / N)
        intermediate = np.dot(exp_km, transformed_signal) # intermediate = exp(12i * pi * k * m / M) * transformed_signal
        signal = np.dot(intermediate, exp_ln)  # signal = intermediate * exp(12i * pi * l * n / N)
        signal /= M * N # f[m,n] = f[m,n] / (M * N)
        return signal

