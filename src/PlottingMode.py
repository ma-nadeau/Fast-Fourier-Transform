import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Callable
import time
from DiscreteFourierTransform2D import DiscreteFourierTransform2D


class PlottingMode:
    def __init__(
        self,
        folder_path: str = "../Results",
    ):
        self.folder_path = folder_path
        self.time_fft: np.ndarray = np.array([])
        self.time_naive: np.ndarray = np.array([])
        self.fft_std: np.ndarray = np.array([])
        self.naive_std: np.ndarray = np.array([])
        
        self.run_experiment()

    def store_and_compute_runtime(
        self, func: Callable, *args, storing_location: str, iterations: int = 10
    ):
        total_time = []
        for _ in range(iterations):
            time_start = time.time()
            func(*args)
            time_end = time.time()
            total_time.append(time_end - time_start)

        total_time = np.array(total_time)
        average_time = total_time.mean()
        std_time = total_time.std()
        if storing_location == "fft":
            self.time_fft = np.append(self.time_fft, average_time)
            self.fft_std = np.append(self.fft_std, std_time)
            print(
                f"Fast Fourier Transform - Average runtime: {average_time} seconds, Standard deviation: {std_time} seconds"
            )

        elif storing_location == "naive":
            self.time_naive = np.append(self.time_naive, average_time)
            self.naive_std = np.append(self.naive_std, std_time)
            print(
                f"Naive Fourier Transform - Average runtime: {average_time} seconds, Standard deviation: {std_time} seconds"
            )

    def clear_time_array(self):
        self.time = np.array([])

    def fft_method(self, arr):
        # TODO: Implement the FFT method
        return np.fft.fft2(arr)

    def naive_ft_method(self, arr):
        t = DiscreteFourierTransform2D(arr)
        return t.compute_dft_2d()

    def plot_average_runtime(
        self,
        sizes: np.array,
    ):
        # Define error bar colors and transparency
        errorbar_alpha = 0.3
        fft_color = "blue"
        naive_color = "red"

        # Plot FFT runtimes
        plt.errorbar(
            sizes,
            self.time_fft,
            yerr=2 * self.fft_std,
            label="FFT",
            color=fft_color,
            fmt="-o",
            ecolor=fft_color,
            elinewidth=1,
            capsize=3,
            alpha=errorbar_alpha,
        )

        # Plot Naive FT runtimes
        plt.errorbar(
            sizes,
            self.time_naive,
            yerr=2 * self.naive_std,
            label="Naive FT",
            color=naive_color,
            fmt="-o",
            ecolor=naive_color,
            elinewidth=1,
            capsize=3,
            alpha=errorbar_alpha,
        )
        plt.grid()
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        plt.savefig(os.path.join(self.folder_path, "Runtime_Analysis.png"))

    # Run experiments
    def run_experiment(self, power=range(5, 11)):
        # TODO: Here, I assume that there's a type in the document, and it isn't 25 to 210 but 2^5 to 2^10
        sizes = np.array([2**i for i in power])
        for size in sizes:
            array = create_2D_array_of_random_element(size)
            self.store_and_compute_runtime(
                self.fft_method, array, storing_location="fft"
            )
            self.store_and_compute_runtime(
                self.naive_ft_method, array, storing_location="naive"
            )

        self.plot_average_runtime(sizes)
        self.clear_time_array()
        pass


def create_2D_array_of_random_element(size: int) -> np.array:
    """Create 2D arrays of random elements of various sizes (sizes must be square and powers of 2"""
    if (size & (size - 1)) != 0 or size <= 0:
        raise ValueError("Size must be a positive power of 2.")
    return np.random.rand(size, size)
