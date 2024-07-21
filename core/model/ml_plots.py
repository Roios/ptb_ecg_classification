from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report, f1_score, hamming_loss,
                             jaccard_score, multilabel_confusion_matrix, precision_recall_curve, precision_score,
                             recall_score)


def plot_signals(signals: List, sample_rate: int = 100) -> None:
    len_signal = len(signals[0])

    t = list(range(0, int(np.ceil(len(signals[0]) / sample_rate))))

    plt.figure(figsize=(12, 6))
    for idx, signal in enumerate(signals):
        if len(signal) != len_signal:
            raise Exception("All the signals need to have the same len.")
        sns.lineplot(data=signal, label=f"Signal {idx + 1}")

    plt.xticks(list(range(0, len_signal, sample_rate)), labels=t)
    plt.legend()
    plt.xlabel("seconds")
    plt.ylabel("Amplitude")
    plt.show()


def plot_loss_evolution(train_losses: np.ndarray,
                        val_losses: np.ndarray,
                        dst_dir: Path = None,
                        show: bool = True) -> None:
    """ Generate the plot of the loss evolution
    """

    plt.close("all")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    epochs_to_plot = np.arange(1, len(train_losses) + 1)

    ax1.plot(epochs_to_plot, train_losses)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.set_xticks(epochs_to_plot)

    ax2.plot(epochs_to_plot, val_losses)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.set_title('Validation Loss')
    ax2.set_xticks(epochs_to_plot)

    plt.suptitle('Training and Validation Loss')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if dst_dir is not None:
        plt.savefig(str(dst_dir.joinpath("loss_evolution.png")))
    if show:
        plt.show()


def plot_pr_curves(ground_truth: np.ndarray,
                   predictions: np.ndarray,
                   labels_name: List,
                   data_type: str = "val",
                   dst_dir: Path = None,
                   show: bool = True) -> Dict:
    """ Generate the pr curve for each label.

        Args:
            ground_truth (np.ndarray): Ground truth labels
            predictions (np.ndarray): Predicted labels
            labels_name (List): Label names
            data_type (str, optional): Name of the data. Defaults to "val".

        Returns:
            Dict: Best threshold per label
        """
    plt.close("all")
    num_labels = len(labels_name)
    num_cols = 3
    num_rows = (num_labels + num_cols - 1) // num_cols

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))

    # Flatten the axs array if necessary
    axs = axs.flatten()

    thresholds_per_label = dict()
    for i, label_name in enumerate(labels_name):
        precision, recall, thresholds = precision_recall_curve(ground_truth[:, i], predictions[:, i])
        average_precision = average_precision_score(ground_truth[:, i], predictions[:, i])

        f1_scores = []
        for p, r in zip(precision, recall):
            if p + r == 0:
                f1_scores.append(np.nan)
            else:
                f1_scores.append(2 * (p * r) / (p + r))
        f1_scores = np.array(f1_scores)
        best_index = np.argmax(f1_scores)
        if np.isnan(f1_scores[best_index]):
            f1_scores[best_index] = -1
            best_index = np.argmax(f1_scores)

        best_threshold = thresholds[best_index] if best_index < len(thresholds) else thresholds[-1]
        best_precision = precision[best_index]
        best_recall = recall[best_index]

        axs[i].plot(recall, precision, label=f'{label_name} (AP={average_precision:.2f})')
        axs[i].scatter([best_recall], [best_precision],
                       marker='o',
                       color='red',
                       label=f'Best (F1={f1_scores[best_index]:.2f} at t={best_threshold:.2f})')
        axs[i].set_ylim([0, 1])
        axs[i].set_xlabel('Recall')
        axs[i].set_ylabel('Precision')
        axs[i].set_title(f'Precision-Recall curve for {label_name}')
        axs[i].legend()

        thresholds_per_label[label_name] = best_threshold

    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    if dst_dir is not None:
        plt.savefig(str(dst_dir.joinpath(f"{data_type}_pr_curves.png")))
    if show:
        plt.show()

    return thresholds_per_label


def plot_cm_multilabel(ground_truth: np.ndarray,
                       predictions: np.ndarray,
                       labels_name: List,
                       data_type: str = "val",
                       dst_dir: Path = None,
                       show: bool = True):
    """ Generate the confusion matrix plot for each label.

    Args:
        ground_truth (np.ndarray): Ground truth labels
        predictions (np.ndarray): Predicted labels
        labels_name (List): Label names
        data_type (str, optional): Name of the data. Defaults to "val".
    """
    plt.close("all")
    confusion_matrices = multilabel_confusion_matrix(ground_truth, predictions)

    num_labels = len(labels_name)
    num_cols = 3
    num_rows = (num_labels + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    axes = axes.flatten()

    for idx, (cm, label) in enumerate(zip(confusion_matrices, labels_name)):
        cm_df = pd.DataFrame(cm,
                             index=[f"True {label}", f"True Not {label}"],
                             columns=[f"Pred {label}", f"Pred Not {label}"])

        cm_percentages = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        annotations = np.array(
            [["{0:.0f}\n({1:.1f}%)".format(cm[i, j], cm_percentages[i, j])
              for j in range(cm.shape[1])]
             for i in range(cm.shape[0])])

        sns.heatmap(cm_df, annot=annotations, fmt='', cmap='Blues', cbar=False, ax=axes[idx], annot_kws={"size": 14})
        axes[idx].set_title(f"Confusion Matrix for {label}")
        axes[idx].set_xlabel("Predicted Labels")
        axes[idx].set_ylabel("True Labels")

    for i in range(len(labels_name), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    if dst_dir is not None:
        plt.savefig(str(dst_dir.joinpath(f"{data_type}_confusion_matrices.png")))
    if show:
        plt.show()


def plot_lr_evolution(lrs: np.ndarray, dst_dir: Path = None, show: bool = True) -> None:
    plt.close("all")
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=lrs, color='black')
    plt.xticks(list(range(1, len(lrs) + 1)))
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Evolution')

    if dst_dir is not None:
        plt.savefig(str(dst_dir.joinpath("lr_evolution.png")))
    if show:
        plt.show()


# @staticmethod
# def report(y_true, y_pred, labels_name):

#     # micro: averages metrics across all classes, emphasizing overall performance
#     # macro: averages metrics independently for each class, giving equal weight to each class
#     accuracy = accuracy_score(y_true, y_pred)

#     # proportion of predicted positive cases that are actually positive across all classes
#     precision_micro = precision_score(y_true, y_pred, average='micro')
#     precision_macro = precision_score(y_true, y_pred, average='macro')
#     precision_weighted = precision_score(y_true, y_pred, average='weighted')
#     # proportion of actual positive cases that are correctly predicted as positive across all classes
#     recall_micro = recall_score(y_true, y_pred, average='micro')
#     recall_macro = recall_score(y_true, y_pred, average='macro')
#     recall_weighted = recall_score(y_true, y_pred, average='weighted')

#     f1_micro = f1_score(y_true, y_pred, average='micro')
#     f1_macro = f1_score(y_true, y_pred, average='macro')
#     f1_weighted = f1_score(y_true, y_pred, average='weighted')

#     # measures the fraction of labels that are incorrectly predicted
#     h_loss = hamming_loss(y_true, y_pred)

#     # measures similarity between the predicted and true label sets
#     jaccard = jaccard_score(y_true, y_pred, average='samples')

#     class_report = classification_report(y_true, y_pred, target_names=labels_name, zero_division=0)

#     print("Overall Metrics:")
#     print(f"Accuracy: {accuracy:.3f} (higher is better)")
#     print(f"Precision (Micro): {precision_micro:.3f} (higher is better)")
#     print(f"Precision (Macro): {precision_macro:.3f} (higher is better)")
#     print(f"Precision (Weighted): {precision_weighted:.3f} (higher is better)")
#     print(f"Recall (Micro): {recall_micro:.3f} (higher is better)")
#     print(f"Recall (Macro): {recall_macro:.3f} (higher is better)")
#     print(f"Recall (Weighted): {recall_weighted:.3f} (higher is better)")
#     print(f"F1-Score (Micro): {f1_micro:.3f} (higher is better)")
#     print(f"F1-Score (Macro): {f1_macro:.3f} (higher is better)")
#     print(f"F1-Score (Weighted): {f1_weighted:.3f} (higher is better)")
#     print(f"Hamming Loss: {h_loss:.3f} (lower is better)")
#     print(f"Jaccard Score: {jaccard:.3f} (higher is better)\n")

#     print("\nClassification Report:")
#     print(f"{class_report}")
