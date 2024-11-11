import numpy as np

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
        k =  np.arange(N // 2)
        # Compute Twiddle factors
        factor = np.exp(-2j * np.pi * k / N)
        
        # for k in range(N // 2):
        # X[k] = X_even[k] + factor[k] * X_odd[k]
        even = X_even + factor * X_odd
        #    X[k + N // 2] = X_even[k] - factor[k] * X_odd[k]
        odd = X_even - factor * X_odd
        
        # Concatenate the even and odd indices
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
            result[k + N // 2] = p - q
        return result

    def fft_2D(self, X: np.ndarray) -> np.ndarray:
        """Compute the 2D Fast Fourier Transform."""
        
        # TODO: Fix size :)
        M, N = X.shape
        print(M, N)
        M_padded = 2**int(np.ceil(np.log2(M)))
        N_padded = 2**int(np.ceil(np.log2(N)))
        X_padded = np.zeros((M_padded, N_padded), dtype=complex)
        X_padded[:M, :N] = X
        X = X_padded
        M, N = X.shape
        print(M, N)
        result = np.zeros((M, N), dtype=complex)

        for m in range(M):
            result[m, :] = self.fft_1D_matrix(X[m, :])

        for n in range(N):
            result[:, n] = self.fft_1D_matrix(result[:, n])

        return result
