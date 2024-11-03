import numpy as np


class DiscreteFourierTransform:
    def __init__(self, signal):
        self.signal = np.array(signal)
        self.transformed_signal = None

    def compute_dft(self):
        """
        Computes the Discrete Fourier Transform (DFT) of the input signal.

        This method transforms a 1D time-domain signal into its frequency-domain representation.

        Returns:
            np.ndarray: The transformed signal in the frequency domain.
        """
        N = len(self.signal)
        self.transformed_signal = np.zeros(N, dtype=complex)
        # for k = 0, 1, 2, ..., N-1
        for k in range(N):
            # for n = 0, 1, 2, ..., N-1
            for n in range(N):
                # X[k] = sum(x[n] * exp( -2i * pi * k * n / N))
                self.transformed_signal[k] += self.signal[n] * np.exp(
                    -2j * np.pi * k * n / N
                )
        return self.transformed_signal

    def compute_inverse_dft(self):
        """
        Computes the Inverse Discrete Fourier Transform (IDFT) of the transformed signal.

        This method reconstructs the original time-domain signal from its frequency-domain representation.

        Returns:
            np.ndarray: The reconstructed signal in the time domain.
        """
        N = len(self.transformed_signal)
        self.signal = np.zeros(N, dtype=complex)
        # for k = 0, 1, 2, ..., N-1
        for k in range(N):
            # for n = 0, 1, 2, ..., N-1
            for n in range(N):
                # x[n] = sum(X[k] * exp( 2i * pi * k * n / N))
                self.signal[n] += self.transformed_signal[k] * np.exp(
                    2j * np.pi * k * n / N
                )
        # x[n] = x[n] / N
        self.signal /= N
        return self.signal

    def get_signal(self):
        return self.signal

    def get_transformed_signal(self):
        return self.transformed_signal
