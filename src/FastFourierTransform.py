import numpy as np
from DiscreteFourierTransform import DiscreteFourierTransform


class FastFourierTransform:
    def __init__(self):
        pass

    def fft_1D_matrix(self, X: np.ndarray) -> np.ndarray:
        """Compute the 1D Fast Fourier Transform using the Cooley-Tukey algorithm."""
        N = X.size
        if N <= 1:
            return X
        # Split the signal into even and odd indices and Recursively compute FFT
        X_even = self.fft_1D_matrix(X[0::2])  # Start at 0 and increment by 2
        X_odd = self.fft_1D_matrix(X[1::2])  # Start at 1 and increment by 2

        # Create a vector for such that k = 0, 1, 2, ..., N/2-1
        k = np.arange(N // 2)

        # Compute Twiddle factors
        # factor = [[exp(-2j * pi * 0 / N)], [exp(-2j * pi * 1 / N)], ..., [exp(-2j * pi * (N/2-1) / N)]]
        factor = np.exp(-2j * np.pi * k / N)

        # X[k] = X_even[k] + factor[k] * X_odd[k]
        # even = [X_even[0] + factor[0] * X_odd[0], X_even[1] + factor[1] * X_odd[1], ..., X_even[N/2-1] + factor[N/2-1] * X_odd[N/2-1]]
        even = X_even + factor * X_odd

        # X[k + N // 2] = X_even[k] - factor[k] * X_odd[k]
        # odd = [X_even[0] - factor[0] * X_odd[0], X_even[1] - factor[1] * X_odd[1], ..., X_even[N/2-1] - factor[N/2-1] * X_odd[N/2-1]]
        odd = X_even - factor * X_odd

        # Concatenate the even and odd indices
        # result = [X[0], X[1], ..., X[N/2-1], X[N/2], X[N/2+1], ..., X[N-1]]
        result = np.concatenate([even, odd])

        return result

    def fft_1D_matrix_dft_when_small(self, X: np.ndarray) -> np.ndarray:
        """Compute the 1D Fast Fourier Transform using the Cooley-Tukey algorithm."""

        N = X.size
        if N <= 1:
            return X

        # If the signal is small, use the Discrete Fourier Transform
        if N <= 1000:  
            dft = DiscreteFourierTransform()
            return dft.dft_1D(X)
        
        # Split the signal into even and odd indices and Recursively compute FFT
        X_even = self.fft_1D_matrix_dft_when_small(X[0::2])  # Start at 0 and increment by 2
        X_odd = self.fft_1D_matrix_dft_when_small(X[1::2])  # Start at 1 and increment by 2

        # Create a vector for such that k = 0, 1, 2, ..., N/2-1
        k = np.arange(N // 2)

        # Compute Twiddle factors
        # factor = [[exp(-2j * pi * 0 / N)], [exp(-2j * pi * 1 / N)], ..., [exp(-2j * pi * (N/2-1) / N)]]
        factor = np.exp(-2j * np.pi * k / N)

        # X[k] = X_even[k] + factor[k] * X_odd[k]
        # even = [X_even[0] + factor[0] * X_odd[0], X_even[1] + factor[1] * X_odd[1], ..., X_even[N/2-1] + factor[N/2-1] * X_odd[N/2-1]]
        even = X_even + factor * X_odd

        # X[k + N // 2] = X_even[k] - factor[k] * X_odd[k]
        # odd = [X_even[0] - factor[0] * X_odd[0], X_even[1] - factor[1] * X_odd[1], ..., X_even[N/2-1] - factor[N/2-1] * X_odd[N/2-1]]
        odd = X_even - factor * X_odd

        # Concatenate the even and odd indices
        # result = [X[0], X[1], ..., X[N/2-1], X[N/2], X[N/2+1], ..., X[N-1]]
        result = np.concatenate([even, odd])
        return result

    def fft_1D_loop(self, X: np.ndarray) -> np.ndarray:
        """Compute the 1D Fast Fourier Transform using the Cooley-Tukey algorithm."""
        N = X.size
        if N <= 1:
            return X

        # Split the signal into even and odd indices and Recursively compute FFT
        X_even = self.fft_1D_loop(X[0::2])  # Start at 0 and increment by 2
        X_odd = self.fft_1D_loop(X[1::2])  # Start at 1 and increment by 2

        result = np.zeros(N, dtype=complex)

        # Here we know that N/2 -1 is the last index of X_even
        for k in range(N // 2):  # for k = 0, 1, 2, ..., N/2-1

            p = X_even[k]
            q = np.exp(-2j * np.pi * k / N) * X_odd[k]
            result[k] = p + q
            #
            result[k + N // 2] = p - q
        return result

    def fft_2D(self, X: np.ndarray) -> np.ndarray:
        """Compute the 2D Fast Fourier Transform."""

        # TODO: Fix size of the matrix to evaluate so that sides of the matrix are 2^n
        M, N = X.shape
        # print(M, N)
        M_padded = 2 ** int(np.ceil(np.log2(M)))
        N_padded = 2 ** int(np.ceil(np.log2(N)))
        X_padded = np.zeros((M_padded, N_padded), dtype=complex)
        X_padded[:M, :N] = X
        X = X_padded
        M, N = X.shape
        # print(M, N)

        result = np.zeros((M, N), dtype=complex)

        # axis = 1 means apply the function along the rows
        intermediate = np.apply_along_axis(
            self.fft_1D_matrix_dft_when_small, axis=1, arr=X
        )

        # axis = 0 means apply the function along the columns
        result = np.apply_along_axis(
            self.fft_1D_matrix_dft_when_small, axis=0, arr=intermediate
        )

        return result
