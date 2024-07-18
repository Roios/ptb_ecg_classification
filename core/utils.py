from collections import Counter
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


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
