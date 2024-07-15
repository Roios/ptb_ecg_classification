from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DST_DIR = Path(__file__).parent.parent.joinpath('images')
CHANNELS_NAME = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
SAMPLING_RATE = 100


def plot_ecg_channels(raw_data: np.ndarray,
                      sampling_rate: int = SAMPLING_RATE,
                      title: str = "ECG example all channels",
                      dst_dir: Path = DST_DIR) -> None:
    """ Plot an ECG.

    Args:
        raw_data (np.ndarray): Signal to plot
        sampling_Rate (int, optional): Sampling rate of the data. Defaults to SAMPLING_RATE
        title (str, optional): Image title
        dst_dir (Path, optional): Where to store the image. Defaults to DST_DIR
    """
    if not dst_dir.exists():
        dst_dir.mkdir(parents=True, exist_ok=True)

    #  a plot per channel
    num_plots = raw_data.shape[1]
    num_cols = 3 if num_plots >= 3 else 1
    num_rows = (num_plots + 1) // num_cols

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(9 * num_cols, 4 * num_rows))
    axs = axs.flatten()

    for i in range(num_plots):
        data_to_plot = raw_data[:, i]
        x_scale = list(range(0, int(np.ceil(len(data_to_plot) / sampling_rate))))
        sns.lineplot(data=data_to_plot, ax=axs[i], linewidth=0.5)
        axs[i].set(xlabel='seconds', ylabel='Amplitude', title=(f'{CHANNELS_NAME[i]}'))
        axs[i].set_xticks(list(range(0, len(data_to_plot), sampling_rate)))
        axs[i].set_xticklabels(x_scale)

    for i in range(num_plots, len(axs)):
        fig.delaxes(axs[i])

    sns.set_theme(font_scale=0.7, style='darkgrid')
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    plt.savefig(str(dst_dir.joinpath(f"{title.lower().replace(' ','_').replace(':','_')}.png")))
    plt.show()


def plot_filtered_signal(ecg_signal: np.ndarray,
                         smoothed_ecg: np.ndarray,
                         sampling_rate: int = SAMPLING_RATE,
                         title: str = "Filtered signal",
                         dst_dir: Path = DST_DIR) -> None:
    """ Plots the original vs the smoothed signal.

    Args:
        ecg_signal (np.ndarray): Original signal to plot
        smoothed_ecg (np.ndarray): Smoothed signal to plot
        sampling_Rate (int, optional): Sampling rate of the data. Defaults to SAMPLING_RATE
        title (str, optional): Title of the plot
        dst_dir (Path, optional): Where to store the image. Defaults to DST_DIR

    """

    t = list(range(0, int(np.ceil(len(ecg_signal) / sampling_rate))))

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=ecg_signal, label="Original", linewidth=0.5)
    sns.lineplot(data=smoothed_ecg, label="Smoothed", color="red", linewidth=0.5)
    plt.xticks(list(range(0, len(ecg_signal), sampling_rate)), labels=t)
    plt.legend()
    plt.xlabel("seconds")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.savefig(str(dst_dir.joinpath(f"{title.replace(' ', '_')}.png")))
    plt.show()


def plot_filtered_signals(ecg_signal: np.ndarray,
                          smoothed_ecgs: List[np.ndarray],
                          labels: List[str],
                          sampling_rate: int = SAMPLING_RATE,
                          title: str = "Filtered signals",
                          dst_dir: Path = DST_DIR) -> None:
    """ Plots the original vs multiple smoothed signal.

    Args:
        ecg_signal (np.ndarray): Original signal to plot
        smoothed_ecgs (List[np.ndarray]): Smoothed signals to plot
        labels (List[str]): Labels of the smooth signals
        sampling_Rate (int, optional): Sampling rate of the data. Defaults to SAMPLING_RATE
        title (str, optional): Title of the plot
        dst_dir (Path, optional): Where to store the image. Defaults to DST_DIR

    """

    t = list(range(0, int(np.ceil(len(ecg_signal) / sampling_rate))))

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=ecg_signal, label="Original", linewidth=0.5)
    for smoothed_ecg, label in zip(smoothed_ecgs, labels):
        sns.lineplot(data=smoothed_ecg, label=label, linewidth=0.5)

    plt.xticks(list(range(0, len(ecg_signal), sampling_rate)), labels=t)
    plt.legend()
    plt.xlabel("seconds")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.savefig(str(dst_dir.joinpath(f"{title.replace(' ', '_')}.png")))
    plt.show()
