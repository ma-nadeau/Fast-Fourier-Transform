import numpy as np


class DiscreteFourierTransform2D:
    def __init__(self, signal):
        self.signal = np.array(signal)
        self.transformed_signal = None

    def compute_dft_2d(self):
        """
        Computes the 2D Discrete Fourier Transform (DFT) of the input signal.

        This method transforms the spatial domain signal into its frequency domain representation.

        Returns:
            np.ndarray: The transformed 2D signal in the frequency domain.
        """
        M, N = self.signal.shape
        self.transformed_signal = np.zeros((M, N), dtype=complex)
        # for u = 0, 1, 2, ..., M-1
        for m in range(M):
            # for v = 0, 1, 2, ..., N-1
            for n in range(N):
                sum = 0
                # for k = 0, 1, 2, ..., M-1
                for k in range(M):
                    # for l = 0, 1, 2, ..., N-1
                    for l in range(N):
                        # F[k,l] = sum (sum f[m,n] * exp(-2i * pi * k * m / M) * exp(-2i * pi * l * n / N)
                        sum += (
                            self.signal[m, n]  # f[m,n]
                            * np.exp(
                                -2j * np.pi * k * m / M
                            )  # exp(-2i * pi * k * m / M)
                            * np.exp(
                                -2j * np.pi * l * n / N
                            )  # exp(-2i * pi * l * n / N)
                        )
                self.transformed_signal[k, l] += sum
        return self.transformed_signal

    def compute_inverse_dft_2d(self):
        """
        Computes the 2D Inverse Discrete Fourier Transform (IDFT) of the transformed signal.

        This method reconstructs the original time domain signal from its frequency domain representation.

        Returns:
            np.ndarray: The reconstructed 2D signal in the time domain.
        """
        M, N = self.transformed_signal.shape
        self.signal = np.zeros((M, N), dtype=complex)

        # for m = 0, 1, 2, ..., M-1
        for m in range(M):
            # for n = 0, 1, 2, ..., N-1
            for n in range(N):
                sum_value = 0
                # for k = 0, 1, 2, ..., M-1
                for k in range(M):
                    # for l = 0, 1, 2, ..., N-1
                    for l in range(N):
                        # f[m,n] = sum (sum F[k,l] * exp(2i * pi * k * m / M) * exp(2i * pi * l * n / N)
                        sum_value += (
                            self.transformed_signal[k, l]  # F[k,l]
                            * np.exp(12 * np.pi * k * m / M)  # exp(2i * pi * k * m / M)
                            * np.exp(12 * np.pi * l * n / N)  # exp(2i * pi * l * n / N)
                        )
                self.signal[m, n] = sum_value
        # f[m,n] = f[m,n] / M * N
        self.signal /= M * N
        return self.signal

    def get_signal(self):
        return self.signal

    def get_transformed_signal(self):
        return self.transformed_signal
