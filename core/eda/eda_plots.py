from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter

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
        sns.lineplot(data=data_to_plot, color='black', ax=axs[i], linewidth=0.5)
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
    sns.lineplot(data=ecg_signal, label="Original", color='black', linewidth=0.5)
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


def plot_signal(signal: np.ndarray, xlabel: str = "", ylabel: str = "", title: str = "") -> None:
    """ Basic line plot of a signal

    Args:
        signal (np.ndarray): Signal to plot
        xlabel (str, optional): x label. Defaults to "".
        ylabel (str, optional): y label. Defaults to "".
        title (str, optional): title. Defaults to "".
    """
    plt.figure(figsize=(20, 4))
    sns.lineplot(data=signal, color='black', linewidth=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def plot_signal_and_rpeaks(signal: np.ndarray,
                           rpeaks_loc: np.ndarray,
                           xlabel: str = "",
                           ylabel: str = "",
                           title: str = "") -> None:
    """ Basic line plot of a signal with the peaks,

    Args:
        signal (np.ndarray): Signal to plot
        rpeaks_loc (np.ndarray): R peaks to plot
        xlabel (str, optional): x label. Defaults to "".
        ylabel (str, optional): y label. Defaults to "".
        title (str, optional): title. Defaults to "".
    """
    plt.figure(figsize=(20, 4))
    sns.lineplot(data=signal, linewidth=0.7, color='black')
    plt.scatter(rpeaks_loc, signal[rpeaks_loc], color='red', s=50, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def plot_data_distribution(class_counts: Dict[str, Dict], class_percentages: Dict[str, Dict]) -> None:
    """ Generate the plot for data distribution.

    Args:
        class_counts (Dict[str, Dict]): Dictionary containing counts per class for each split.
        class_percentages (Dict[str, Dict]): Dictionary containing percentages per class for each split.
    """
    splits = list(class_counts.keys())
    num_splits = len(splits)

    _, axes = plt.subplots(2, num_splits, figsize=(5 * num_splits, 10))
    plt.subplots_adjust(hspace=0.5)

    for i, split in enumerate(splits):
        df = pd.DataFrame({
            'Class': list(class_counts[split].keys()),
            f'{split}_Counts': list(class_counts[split].values()),
            f'{split}_Percentage': [class_percentages[split].get(cls, 0) for cls in class_counts[split].keys()]
        }).sort_values(by='Class')

        sns.barplot(ax=axes[0, i], x='Class', y=f'{split}_Percentage', data=df)
        axes[0, i].set_title(f'{split} Class Distribution (Percentage)')
        axes[0, i].set_ylabel('Percentage' if i == 0 else '')
        axes[0, i].set_xticks(range(len(df)))
        axes[0, i].set_xticklabels(df['Class'], rotation=90)

        sns.barplot(ax=axes[1, i], x='Class', y=f'{split}_Counts', data=df)
        axes[1, i].set_title(f'{split} Class Distribution (Counts)')
        axes[1, i].set_ylabel('Counts' if i == 0 else '')
        axes[1, i].set_xticks(range(len(df)))
        axes[1, i].set_xticklabels(df['Class'], rotation=90)

    plt.suptitle("Data distribution")
    plt.tight_layout()
    plt.show()


def plot_distribution_age_sex(annotations) -> None:
    sns.histplot(data=annotations, x='age', hue="sex", binwidth=5)
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.title('Age distribution by sex')
    plt.show()


def plot_distribution_diagnostic_sex(annotations) -> None:
    aux_dict = {"sex": [], "diagnostic": []}
    for diagnostic, sex in zip(annotations.diagnostic_superclass, annotations.sex):
        aux_dict['diagnostic'].append(" ".join(l for l in diagnostic))
        aux_dict['sex'].append(sex)

    df = pd.DataFrame(aux_dict)
    df = df.sort_values(by='diagnostic')

    count_df = df.groupby(['sex', 'diagnostic']).size().reset_index(name='count')
    total_counts = count_df.groupby('sex')['count'].transform('sum')
    count_df['percentage'] = (count_df['count'] / total_counts) * 100

    _, axs = plt.subplots(2, 1, figsize=(10, 8))
    sns.barplot(ax=axs[0],
                data=count_df,
                x='diagnostic',
                y='count',
                hue='sex',
                palette='Set2',
                order=sorted(df['diagnostic'].unique()))
    axs[0].set_xlabel('Diagnostic')
    axs[0].set_ylabel('Count')
    axs[0].set_title('Counts of diagnostic by sex')
    axs[0].tick_params(axis='x', rotation=90)

    sns.barplot(ax=axs[1],
                data=count_df,
                x='diagnostic',
                y='percentage',
                hue='sex',
                palette='Set2',
                order=sorted(df['diagnostic'].unique()))
    axs[1].yaxis.set_major_formatter(PercentFormatter())
    axs[1].set_xlabel('Diagnostic')
    axs[1].set_ylabel('Percentage')
    axs[1].set_title('Percentages of diagnostic by sex')
    axs[1].tick_params(axis='x', rotation=90)

    plt.tight_layout()
    plt.show()
