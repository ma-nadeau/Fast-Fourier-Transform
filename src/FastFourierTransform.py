import numpy as np

class FastFourierTransform:
    def __init__(self):
        pass

    def fft_1D(self, X: np.ndarray) -> np.ndarray:
        """Compute the 1D Fast Fourier Transform."""
        N = X.size
        if N == 1:
            return X

        # Split the signal into even and odd indices
        X_even = self.fft_1D(X[0::2])  # Start at 0 and increment by 2
        X_odd = self.fft_1D(X[1::2])  # Start at 1 and increment by 2

        # Result placeholder
        result = np.zeros(N, dtype=complex)

        # Here we know that N/2 -1 is the last index of X_even
        for k in range( int(N / 2 - 1)):  # for k = 0, 1, 2, ..., N/2-1
            p = X_even[k]
            q = np.exp(-2j * np.pi * k / N) * X_odd[k]
            result[k] = p + q
            result[k + N // 2] = p - q
        return result

    def fft_2D(self, X: np.ndarray) -> np.ndarray:
        """Compute the 2D Fast Fourier Transform."""
        M, N = X.shape
        result = np.zeros((M, N), dtype=complex)

        for m in range(M):
            result[m, :] = self.fft_1D(X[m, :])

        for n in range(N):
            result[:, n] = self.fft_1D(result[:, n])

        return result
