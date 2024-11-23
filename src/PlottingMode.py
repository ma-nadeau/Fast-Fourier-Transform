import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List
import time
from FourierTransform import transform_2D


class PlottingMode:
    """ Class responsible for comparing and plotting runtime results
     for the 2D DFT and 2D FFT algorithms """

    def __init__(self, powers_2: List[int] = list(range(5, 11))):
        self.fft_means: np.ndarray = np.array([])
        self.naive_means: np.ndarray = np.array([])
        self.fft_stds: np.ndarray = np.array([])
        self.naive_stds: np.ndarray = np.array([])

        self.run_experiment(powers_2)

    def run_experiment(self, powers_2: List[int]) -> None:
        # Clear the time arrays in case the experiment was run more than once
        self.clear_time_array()

        for power_2 in powers_2:
            self.compute_runtime_estimate(power_2, isFFT=True)
            self.compute_runtime_estimate(power_2, isFFT=False)

        self.powers_2 = powers_2

    def compute_runtime_estimate(
        self, power_2: int, isFFT: bool, iterations: int = 10
    ) -> None:
        samples = []
        array = create_2D_array_of_random_element(power_2)
        for _ in range(iterations):
            time_start = time.time()
            transform_2D(array) if isFFT else transform_2D(array, False)
            time_end = time.time()
            samples.append(time_end - time_start)

        samples = np.array(samples)
        average_time = samples.mean()
        std_time = samples.std()
        results = f"Average runtime: {average_time:.2f} seconds, Standard deviation: {std_time:.2f} seconds, Array size: {2**power_2}"
        if isFFT:
            self.fft_means = np.append(self.fft_means, average_time)
            self.fft_stds = np.append(self.fft_stds, std_time)
            print(f"Fast Fourier Transform - {results}")
        else:
            self.naive_means = np.append(self.naive_means, average_time)
            self.naive_stds = np.append(self.naive_stds, std_time)
            print(f"Naive Fourier Transform - {results}")

    def clear_time_array(self) -> None:
        self.fft_means = np.array([])
        self.naive_means = np.array([])
        self.fft_stds = np.array([])
        self.naive_stds = np.array([])

    def plot_average_runtime(self, save_plot: bool = False) -> None:

        sizes = 2 ** np.array(self.powers_2)

        plt.suptitle("Runtime of FFT", fontsize=18, fontweight="bold")

        # Define error bar colors and transparency
        errorbar_alpha = 0.3
        fft_color = "blue"
        naive_color = "red"

        # Plot FFT runtimes
        plt.errorbar(
            sizes,
            self.fft_means,
            yerr=2 * self.fft_stds,
            label="Fast Fourier Transform",
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
            self.naive_means,
            yerr=2 * self.naive_stds,
            label="Naive Fourier Transform",
            color=naive_color,
            fmt="-o",
            ecolor=naive_color,
            elinewidth=1,
            capsize=3,
            alpha=errorbar_alpha,
        )
        plt.legend()
        plt.xlabel("Matrix Size", fontweight="bold")
        plt.ylabel("Time (s)", fontweight="bold")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        if save_plot:
            folder_path = "../Results"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            plt.savefig(os.path.join(folder_path, "Runtime_Analysis.png"))
        plt.show()


def create_2D_array_of_random_element(power_2: int) -> np.array:
    """Create 2D arrays of random elements of various sizes (sizes must be square and powers of 2"""
    size = 2**power_2
    return np.random.rand(size, size)
