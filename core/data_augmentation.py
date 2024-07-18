from typing import List, Union

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from core.utils import binarize_labels


class DataAugmentor():

    def get_undersampled(self, data: np.ndarray, labels: pd.Series, add_multi: bool = True) -> Union[np.ndarray, List]:
        """ Under sample the data.

        Args:
            data (np.ndarray): data
            labels (pd.Series): labels
            add_multi (bool, optional): Add multi label cases to resampled data. Defaults to True.

        Returns:
            Union[np.ndarray, List]: Resampled data and labels
        """
        # Get only data with single label
        single_label_data, single_labels = self.get_single_label_only(data, labels)

        # Binarize the labels
        labels_train_bin, labels_class = binarize_labels(single_labels)

        # Under sample the data
        data_resampled_under, labels_resampled_under = self.undersample_data(single_label_data, labels_train_bin,
                                                                             labels_class)

        # Rework the labels format
        new_labels = [[labels_class[l]] for l in np.argmax(labels_resampled_under, axis=1)]

        if add_multi:
            data_resampled_under, new_labels = self.add_multilabel_cases(data, labels, data_resampled_under, new_labels)
        return data_resampled_under, new_labels

    def get_oversampled(self, data: np.ndarray, labels: pd.Series, add_multi: bool = True) -> Union[np.ndarray, List]:
        """ Over sample the data.

        Args:
            data (np.ndarray): data
            labels (pd.Series): labels
            add_multi (bool, optional): Add multi label cases to resampled data. Defaults to True.

        Returns:
            Union[np.ndarray, List]: Resampled data and labels
        """
        # Get only data with single label
        single_label_data, single_labels = self.get_single_label_only(data, labels)

        # Binarize the labels
        labels_train_bin, labels_class = binarize_labels(single_labels)

        # Over sample the data
        data_resampled_over, labels_resampled_over = self.oversample_data(single_label_data, labels_train_bin,
                                                                          labels_class)

        # Rework the labels format
        new_labels = [[labels_class[l]] for l in np.argmax(labels_resampled_over, axis=1)]

        if add_multi:
            data_resampled_over, new_labels = self.add_multilabel_cases(data, labels, data_resampled_over, new_labels)
        return data_resampled_over, new_labels

    def add_multilabel_cases(self, data: np.ndarray, labels: pd.Series, resampled_data: np.ndarray,
                             resampled_labels: List) -> Union[np.ndarray, List]:
        """ Add the multi labeled cases to resampled data.

        Args:
            data (np.ndarray): original data
            labels (pd.Series): original labels
            resampled_data (np.ndarray): resampled data
            resampled_labels (List): resampled labels

        Raises:
            Exception: Sizes not matching

        Returns:
            Union[np.ndarray, List]: Combined data and labels
        """
        # Get multilabel entries only
        multi_label_data, multi_labels = self.get_multi_label_only(data, labels)
        multi_labels = multi_labels.to_list()
        # Add the resampled data to the multilabeled cases
        multi_label_data = np.concatenate([multi_label_data, resampled_data])
        multi_labels.extend(resampled_labels)
        if multi_label_data.shape[0] != len(multi_labels):
            raise Exception("The dims from labels and data don't match.")

        return multi_label_data, multi_labels

    @staticmethod
    def get_single_label_only(data: np.ndarray, labels: pd.Series) -> Union[np.ndarray, pd.Series]:
        """ Get only the entry points that are single labeled.

        Args:
            data (np.ndarray): data
            labels (pd.Series): labels

        Returns:
            Union[np.ndarray, pd.Series]: target data and labels
        """
        data_idx_to_ignore = [idx for idx, label in enumerate(labels) if len(label) != 1]
        single_label_data = np.delete(data, data_idx_to_ignore, 0)

        label_idx_to_ignore = [labels.index[idx] for idx, label in enumerate(labels) if len(label) != 1]
        single_labels = labels.drop(label_idx_to_ignore)

        return single_label_data, single_labels

    @staticmethod
    def get_multi_label_only(data: np.ndarray, labels: pd.Series) -> Union[np.ndarray, pd.Series]:
        """ Get only the entry points that are multi labeled.

        Args:
            data (np.ndarray): data
            labels (pd.Series): labels

        Returns:
            Union[np.ndarray, pd.Series]: target data and labels
        """
        data_idx_to_ignore = [idx for idx, label in enumerate(labels) if len(label) == 1]
        multi_label_data = np.delete(data, data_idx_to_ignore, 0)

        label_idx_to_ignore = [labels.index[idx] for idx, label in enumerate(labels) if len(label) == 1]
        multi_labels = labels.drop(label_idx_to_ignore)
        return multi_label_data, multi_labels

    @staticmethod
    def undersample_data(data_array: np.ndarray, labels_array: np.ndarray,
                         labels_class: List) -> Union[np.ndarray, np.ndarray]:
        """ Under sample the data based on labels.

        Args:
            data_array (np.ndarray): data to under sample.
            labels_array (np.ndarray): binarized labels to consider.

        Returns:
            Union[np.ndarray, np.ndarray]: Data and labels under sampled
        """
        _, data_len, data_channels = data_array.shape
        _, num_labels = labels_array.shape

        num_samples = data_array.shape[0]
        data_reshaped = data_array.reshape(num_samples, -1)
        labels_reshaped = np.argmax(labels_array, axis=1)

        sampler = RandomUnderSampler(sampling_strategy='auto')
        data_resampled, labels_resampled = sampler.fit_resample(data_reshaped, labels_reshaped)

        data_resampled = data_resampled.reshape(-1, data_len, data_channels)

        unique_labels, counts = np.unique(labels_resampled, return_counts=True)
        label_distribution = {labels_class[l]: c for l, c in zip(unique_labels, counts)}
        print(f"Undersample distribution: {label_distribution}")

        labels_reshaped = np.zeros((labels_resampled.shape[0], num_labels))
        for idx, val in enumerate(labels_resampled):
            labels_reshaped[idx][val] = 1

        return data_resampled, labels_reshaped

    @staticmethod
    def oversample_data(data_array: np.ndarray, labels_array: np.ndarray,
                        labels_class: List) -> Union[np.ndarray, np.ndarray]:
        """ Over sample the data based on labels.

        Args:
            data_array (np.ndarray): data to over sample.
            labels_array (np.ndarray): binarized labels to consider.

        Returns:
            Union[np.ndarray, np.ndarray]: Data and labels over sampled
        """
        _, data_len, data_channels = data_array.shape
        _, num_labels = labels_array.shape

        num_samples = data_array.shape[0]
        data_reshaped = data_array.reshape(num_samples, -1)
        labels_reshaped = np.argmax(labels_array, axis=1)

        sampler = RandomOverSampler(sampling_strategy='auto')
        data_resampled, labels_resampled = sampler.fit_resample(data_reshaped, labels_reshaped)

        data_resampled = data_resampled.reshape(-1, data_len, data_channels)

        unique_labels, counts = np.unique(labels_resampled, return_counts=True)
        label_distribution = {labels_class[l]: c for l, c in zip(unique_labels, counts)}
        print(f"Oversample distribution: {label_distribution}")

        labels_reshaped = np.zeros((labels_resampled.shape[0], num_labels))
        for idx, val in enumerate(labels_resampled):
            labels_reshaped[idx][val] = 1

        return data_resampled, labels_reshaped
