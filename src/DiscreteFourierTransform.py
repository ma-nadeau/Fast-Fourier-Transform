import numpy as np


def dft_naive_1D(signal : np.ndarray, inverse : bool = False) -> np.ndarray:
    # Important metrics
    N = signal.shape[0]
    signal_range = np.arange(0,N)
    
    # The idea is to see DFT as matrix multiplication of signal and 
    # Exponent matrix
    exp_array = lambda k : np.exp((1 if inverse else -1) * 2j*np.pi*signal_range*k/N)
    exp_matrix = np.transpose(np.array([exp_array(k) for k in signal_range]))

    transformed_signal = (signal @ exp_matrix) * (1/N if inverse else 1)

    return np.array((transformed_signal).reshape(transformed_signal.shape[0]))

def dft_naive_2D(signal : np.ndarray, inverse : bool = False) -> np.ndarray:
    transformed_row_signal = np.apply_along_axis(dft_naive_1D, axis=1, arr=signal, inverse = inverse)
    transformed_signal = np.apply_along_axis(dft_naive_1D, axis=0, arr=transformed_row_signal, inverse = inverse)
    return transformed_signal
