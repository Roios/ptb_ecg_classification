from collections import Counter
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import WeightedRandomSampler


def calculate_distribution(labels: List, use_combo: bool = False) -> Union[Dict, Dict]:
    """ Calculate the labels distributions.

        Args:
            labels (List): labels to analyze
            use_combo (bool, optional): Consider the multi label has 1 label. Defaults to False.

        Returns:
            Union[Dict, Dict]: Labels count and percentages
        """
    if use_combo:
        class_counts = Counter(" ".join(np.sort(l)) for l in labels)
    else:
        class_counts = Counter(ll for l in labels for ll in l)
    total_count = sum(class_counts.values())
    class_percentages = {cls: count / total_count * 100 for cls, count in class_counts.items()}
    return class_counts, class_percentages


def binarize_labels(multi_labels: pd.Series) -> Union[np.ndarray, List]:
    """ Binarize labels.

    Args:
        multi_labels (pd.Series): Labels to binarize

    Returns:
        Union[np.ndarray, List]: Binarized labels and classes
    """
    mlb = MultiLabelBinarizer()
    mlb.fit(multi_labels)
    bin_labels = mlb.transform(multi_labels)
    return bin_labels, mlb.classes_


def get_binarize_transform(multi_labels: pd.Series) -> MultiLabelBinarizer:
    """ Get binarize transform

    Args:
        multi_labels (pd.Series): Labels to compute the transform

    Returns:
        MultiLabelBinarizer: Binarize transform
    """
    mlb = MultiLabelBinarizer()
    mlb.fit(multi_labels)
    return mlb


def remove_no_label_data(data, labels):
    # Remove data points without labels
    def is_invalid_label(x):
        return isinstance(x, list) and len(x) == 0

    data_idx_to_ignore = [idx for idx, label in enumerate(labels) if is_invalid_label(label)]
    signal_filtered = np.delete(data, data_idx_to_ignore, 0)

    label_idx_to_ignore = [labels.index[idx] for idx, label in enumerate(labels) if is_invalid_label(label)]
    labels_filtered = labels.drop(label_idx_to_ignore)

    return signal_filtered, labels_filtered


def get_mean_and_std(signal):
    mu = np.mean(signal, keepdims=True)
    sigma = np.std(signal, keepdims=True)
    return mu, sigma


def get_mean_and_std_per_channel(signal):
    channels = signal.shape[-1]
    mu = [np.mean(signal[:, :, i], keepdims=True) for i in range(channels)]
    sigma = [np.std(signal[:, :, i], keepdims=True) for i in range(channels)]
    return mu, sigma


def standardize_signal(signal, mean, std):
    signal = (signal - mean) / std
    return signal


def standardize_signal_per_channel(signal, mean, std):
    for i in range(signal.shape[-1]):
        signal[:, :, i] = (signal[:, :, i] - mean[i]) / std[i]
    return signal


def balance_input(data: np.ndarray, labels: np.ndarray) -> Union[torch.utils.data.Sampler, torch.tensor]:
    indices = torch.randperm(len(data))
    data = data[indices]
    labels = labels[indices]
    class_counts = labels.sum(dim=0)
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum()

    weights = torch.tensor([class_weights[label] for label in labels.argmax(dim=1)])
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    return sampler, class_weights


class EarlyStopping:

    def __init__(self, patience: int = 7, delta: float = 0, path: str = "checkpoint.pt", verbose: bool = True) -> None:
        """
        Args:
            patience (int): Epochs to wait after last val loss improved.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str, optional): Path for the checkpoint to be saved to. Default to "checkpoint.pt"
            verbose (bool, optional): Verbosity. Default to False.
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.verbose = verbose

    def __call__(self, val_loss: float, model) -> None:

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.verbose:
                print(f"New best score! It was {self.best_score:.6f} now is {score:.6f}")
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model) -> None:
        """ Saves model when validation loss decrease.
        """

        if self.verbose:
            print("Reset counter and saving model...")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

    def load_checkpoint(self, model):
        model.load_state_dict(torch.load(self.path))
        model.eval()
