from collections import Counter
from typing import Dict, List, Union

import numpy as np


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
