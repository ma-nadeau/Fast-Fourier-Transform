import cv2
import numpy as np


def compute_exp_coeffs(signal_size: int, inverse: bool = False):
    """Computes the matrix of exponents coefficients for the DFT and FFT"""

    signal_range = np.arange(signal_size)  # signal_range = [0, 1, 2, ..., N-1]

    # This matrix represents all combinations of k * n in the [i]dft
    matrix_kn = np.outer(signal_range, signal_range)
    # Note
    # k outer n =
    # [[0×0, 0×1, 0×2, ..., 0×(N-1)],
    #  [1×0, 1×1, 1×2, ..., 1×(N-1)],
    #  [2×0, 2×1, 2×2, ..., 2×(N-1)],
    #  ...,
    #  [(N-1)×0, (N-1)×1, (N-1)×2, ..., (N-1)×(N-1)]]

    return np.exp((1 if inverse else -1) * 2j * np.pi * matrix_kn / signal_size)


# Precomputed coefficients for FFT
dft_precomputed_exp_coeffs4 = compute_exp_coeffs(4)
idft_precomputed_exp_coeffs4 = compute_exp_coeffs(4, True)


def dft_naive_1D(signal: np.ndarray) -> np.ndarray:
    """Computes the 1D [Inverse] Discrete Fourier Transform  (DFT) of the input signal.
    This method transforms the spatial domain signal into its frequency domain representation.
    Returns:
        np.ndarray: The transformed 1D signal in the frequency domain.
    """
    # Important metrics
    N = signal.size

    # Matrix of exponent coefficients
    exp_kn = compute_exp_coeffs(N)

    # DFT as a matrix multiplication
    # Each sum can be viewed as a dot product so the whole signal
    # can be obtained by matrix multiplcation (which is essentially many dot products)
    return signal @ exp_kn


def idft_naive_1D(signal: np.ndarray) -> np.ndarray:
    """Computes the 1D [Inverse] Discrete Fourier Transform  (DFT) of the input signal.
    This method transforms the spatial domain signal into its frequency domain representation.
    Returns:
        np.ndarray: The transformed 1D signal in the frequency domain.
    """
    # Important metrics
    N = signal.size

    # Matrix of exponent coefficients
    exp_kn = compute_exp_coeffs(N, True)

    # DFT as a matrix multiplication
    # Each sum can be viewed as a dot product so the whole signal
    # can be obtained by matrix multiplcation (which is essentially many dot products)
    return 1 / N * (signal @ exp_kn)


def fft_1D(signal: np.ndarray) -> np.ndarray:
    """Compute the 1D Fast Fourier Transform using the Cooley-Tukey algorithm."""

    # Important metrics
    N = signal.size

    # If the signal length is too small then just use dft
    if N < 4:
        return dft_naive_1D(signal)
    # If the signal length is 4 then we can leverage precomputed exponents
    elif N == 4:
        return signal @ dft_precomputed_exp_coeffs4

    # Split the signal into even and odd indices and Recursively compute FFT
    X_even = fft_1D(signal[::2])  # Start at 0 and increment by 2
    X_odd = fft_1D(signal[1::2])  # Start at 1 and increment by 2

    # Create a vector for such that k = 0, 1, 2, ..., N/2-1
    k = np.arange(N // 2)

    # Compute Twiddle factors
    factor = np.exp(-2j * np.pi * k / N)

    # for k in range(N // 2):
    # X[k] = X_even[k] + factor[k] * X_odd[k]
    first_half = X_even + factor * X_odd

    #    X[k + N // 2] = X_even[k] - factor[k] * X_odd[k]
    second_half = X_even - factor * X_odd

    # Concatenate the first and second half
    return np.concatenate([first_half, second_half])


def ifft_1D(signal: np.ndarray) -> np.ndarray:
    """Compute the 1D Fast Fourier Transform using the Cooley-Tukey algorithm."""

    # Important metrics
    N = signal.size

    # If the signal length is too small then just use dft
    if N < 4:
        return idft_naive_1D(signal)
    # If the signal length is 4 then we can leverage precomputed exponents
    elif N == 4:
        return 1 / N * (signal @ idft_precomputed_exp_coeffs4)

    # Split the signal into even and odd indices and Recursively compute FFT
    X_even = ifft_1D(signal[::2])  # Start at 0 and increment by 2
    X_odd = ifft_1D(signal[1::2])  # Start at 1 and increment by 2

    # Create a vector for such that k = 0, 1, 2, ..., N/2-1
    k = np.arange(N // 2)

    # Compute Twiddle factors
    factor = np.exp(2j * np.pi * k / N)

    # for k in range(N // 2):
    # X[k] = X_even[k] + factor[k] * X_odd[k]
    first_half = X_even + factor * X_odd

    #    X[k + N // 2] = X_even[k] - factor[k] * X_odd[k]
    second_half = X_even - factor * X_odd

    # Concatenate the first and second half
    return (1 / 2) * np.concatenate([first_half, second_half])


def transform_2D(signal: np.ndarray, fast: bool = True) -> np.ndarray:
    transformed_row_signal = np.apply_along_axis(
        fft_1D if fast else dft_naive_1D, axis=1, arr=signal
    )
    transformed_signal = np.apply_along_axis(
        fft_1D if fast else dft_naive_1D, axis=0, arr=transformed_row_signal
    )
    return transformed_signal


def inverse_transform_2D(signal: np.ndarray, fast: bool = True) -> np.ndarray:
    transformed_row_signal = np.apply_along_axis(
        ifft_1D if fast else idft_naive_1D, axis=1, arr=signal
    )
    transformed_signal = np.apply_along_axis(
        ifft_1D if fast else idft_naive_1D, axis=0, arr=transformed_row_signal
    )
    return transformed_signal


def rescale_image_power2(image):
    old_height, old_width = image.shape
    new_dimensions = (nearest_power2(old_width), nearest_power2(old_height))
    return cv2.resize(image, new_dimensions)


def nearest_power2(n: int):
    return 2 ** int(np.ceil(np.log2(n)))
