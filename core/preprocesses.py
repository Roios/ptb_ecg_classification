# Data processes for cleanup and eda

from collections import Counter
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.signal import butter, lfilter, medfilt, savgol_filter, sosfiltfilt


def smooth_signal_savgol(ecg_signal: np.ndarray, window_length: int, polyorder: int = 3) -> np.ndarray:
    """ Applies the Savitzky-Golay filter to smooth an ECG.

    Args:
        ecg_signal (np.ndarray): ECG to smooth
        window_length (int): The length of the filter window
        polyorder (int, optional): The order of the polynomial. Defaults to 3.

    Returns:
        np.ndarray: The smoothed signal
    """
    smoothed_ecg = savgol_filter(ecg_signal, window_length=window_length, polyorder=polyorder)

    return smoothed_ecg


def smooth_signal_gaussian(ecg_signal: np.ndarray, sigma: int = 2) -> np.ndarray:
    """ Applies the Gaussian filter to smooth an ECG.

    Args:
        ecg_signal (np.ndarray): ECG to smooth
        sigma (int, optional): Standard deviation for Gaussian kernel. Defaults to 2.

    Returns:
        np.ndarray: The smoothed signal
    """
    smoothed_ecg = gaussian_filter(ecg_signal, sigma=sigma)

    return smoothed_ecg


def smooth_signal_median(ecg_signal: np.ndarray, kernel_size: int = 11) -> np.ndarray:
    """ Applies the Median filter to smooth an ECG.

    Args:
        ecg_signal (np.ndarray): ECG to smooth
        kernel_size (int, optional): Size of the median filter window. Defaults to 11.

    Returns:
        np.ndarray: The smoothed signal
    """
    smoothed_ecg = medfilt(ecg_signal, kernel_size=kernel_size)

    return smoothed_ecg


def smooth_signal_lowpass(ecg_signal: np.ndarray,
                          sample_rate: int = 100,
                          order_filter: int = 5,
                          cut: float = 45) -> np.ndarray:
    """ Applies the Low-pass filter to smooth an ECG.

    Args:
        ecg_signal (np.ndarray): ECG to smooth
        sample_rate (int, optional): Sampling rate of the signal. Defaults to 100.
        order_filter (int, optional): Order of the filter. Defaults to 5.
        low (float, optional): Cutoff frequency in Hz. Defaults to 45.

    Returns:
        np.ndarray: The smoothed signal
    """
    out = butter(order_filter, cut, fs=sample_rate, btype='lowpass', output='sos')
    filtered_ecg = sosfiltfilt(out, ecg_signal)

    return filtered_ecg


def smooth_signal_highpass(ecg_signal: np.ndarray,
                           sample_rate: int = 100,
                           order_filter: int = 5,
                           cut: float = 0.5) -> np.ndarray:
    """ Applies the High-pass filter to smooth an ECG.

    Args:
        ecg_signal (np.ndarray): ECG to smooth
        sample_rate (int, optional): Sampling rate of the signal. Defaults to 100.
        order_filter (int, optional): Order of the filter. Defaults to 5.
        low (float, optional): Cutoff frequency in Hz. Defaults to 0.5.

    Returns:
        np.ndarray: The smoothed signal
    """
    out = butter(order_filter, cut, fs=sample_rate, btype='highpass', output='sos')
    filtered_ecg = sosfiltfilt(out, ecg_signal)

    return filtered_ecg


def smooth_signal_butterworth(ecg_signal: np.ndarray,
                              sample_rate: int = 100,
                              order_filter: int = 5,
                              lowcut: float = 0.5,
                              highcut: float = 49.0) -> np.ndarray:
    """ Applies the Butterworth filter to smooth an ECG.

    Args:
        ecg_signal (np.ndarray): ECG to smooth
        sample_rate (int, optional): Sampling rate of the signal. Defaults to 100.
        order_filter (int, optional): Order of the filter. Defaults to 5.
        lowcut (float, optional): Lower cutoff frequency in Hz. Defaults to 0.5.
        highcut (float, optional): Upper cutoff frequency in Hz. Defaults to 49.0.

    Returns:
        np.ndarray: The smoothed signal
    """

    out = butter(order_filter, [lowcut, highcut], fs=sample_rate, btype='bandpass', output='sos')
    filtered_ecg = sosfiltfilt(out, ecg_signal)

    return filtered_ecg


def smooth_signal_convolution(ecg_signal: np.ndarray, kernel: int = 10) -> np.ndarray:
    """ Applies the Convolution filter to smooth an ECG.

    Args:
        ecg_signal (np.ndarray): ECG to smooth
        sample_rate (int, optional): Sampling rate of the signal. Defaults to 100.
        kernel (int, optional): How wide is the filter. Defaults to 10.

    Returns:
        np.ndarray: The smoothed signal
    """

    kernel = np.hanning(kernel)
    if kernel.sum() == 0:
        raise ZeroDivisionError()
    kernel = kernel / kernel.sum()

    filtered_ecg = np.convolve(kernel, ecg_signal, mode='valid')

    return filtered_ecg


def estimate_baseline_wander(ecg_signal: np.ndarray, durations: List[float], sample_rate: int = 100) -> np.ndarray:
    """ Estimate the baseline wander of a signal using accumulative median filters.

    Args:
        ecg_signal (np.ndarray): ECG to estimate the wander from
        durations (List[float]): Window size in seconds for each filter
        sample_rate (int, optional): Sampling rate of the signal. Defaults to 100.

    Returns:
        np.ndarray: baseline wander
    """

    def get_median_filter_width(sample_rate, duration):
        res = int(sample_rate * duration)
        res += ((res % 2) - 1)  # needs to be an odd number
        return res

    wander = ecg_signal
    for duration in durations:
        kernel_size = get_median_filter_width(sample_rate, duration)
        wander = smooth_signal_median(ecg_signal=wander, kernel_size=kernel_size)

    return wander


def remove_baseline_wander(ecg_signal: np.ndarray, durations: List[float], sample_rate: int = 100) -> np.ndarray:
    """ Remove the baseline wander of a signal using accumulative median filters.

    Args:
        ecg_signal (np.ndarray): ECG to estimate and remove the wander from
        durations (List[float]): Window size in seconds for each filter
        sample_rate (int, optional): Sampling rate of the signal. Defaults to 100.

    Returns:
        np.ndarray: signal without the wander
    """
    wander = estimate_baseline_wander(ecg_signal, durations, sample_rate)
    filtered_ecg = np.subtract(ecg_signal, wander)
    return filtered_ecg


def calculate_distribution(labels: List) -> Union[Dict, Dict]:
    """ Calculate the labels distributions.

    Args:
        labels (List): labels to analyze

    Returns:
        Union[Dict, Dict]: Labels count and percentages
    """
    class_counts = Counter(ll for l in labels for ll in l)
    total_count = sum(class_counts.values())
    class_percentages = {cls: count / total_count * 100 for cls, count in class_counts.items()}
    return class_counts, class_percentages
