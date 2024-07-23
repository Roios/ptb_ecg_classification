import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import core.utils as crutils
from core.eda.data_augmentation import DataAugmentor

logger = logging.getLogger(__name__)


class ECGDataset(Dataset):

    def __init__(self,
                 data,
                 apply_sampler: bool = False,
                 ds_type: str = "train",
                 shuffle: bool = False,
                 augmentation: str = None) -> None:

        # Remove data without annotations
        train_signal_clean, train_labels_filtered = crutils.remove_no_label_data(data["train"]["data"],
                                                                                 data["train"]["labels"])
        if ds_type != "train":
            signal_clean, labels_filtered = crutils.remove_no_label_data(data["val"]["data"], data["val"]["labels"])

        # data augmentation
        if ds_type != "train" and augmentation is not None:
            augmentation = None
            logger.info("Augmentation is only valid for training data. Ignoring augmentation request.")

        if augmentation is not None:
            aug = DataAugmentor()
            init_size = train_signal_clean.shape[0]
            if augmentation.lower() == "over":
                train_signal_clean, train_labels_filtered = aug.get_oversampled(train_signal_clean,
                                                                                train_labels_filtered,
                                                                                add_multi=True)
            elif augmentation.lower() == "under":
                train_signal_clean, train_labels_filtered = aug.get_undersampled(train_signal_clean,
                                                                                 train_labels_filtered,
                                                                                 add_multi=True)
            new_size = train_signal_clean.shape[0]

            print(train_signal_clean.shape, type(train_signal_clean))
            print(len(train_labels_filtered), type(train_labels_filtered))
            logger.info(
                f"The training data was augmented. Before we had {init_size} samples and now we have {new_size}")

        # Binarize the labels
        bin_transf = crutils.get_binarize_transform(train_labels_filtered)
        if ds_type != "train":
            labels_bin = bin_transf.transform(labels_filtered)
            labels_class = bin_transf.classes_
        else:
            labels_bin, labels_class = crutils.binarize_labels(train_labels_filtered)

        # Standardize the data per channel
        mean_train, std_train = crutils.get_mean_and_std_per_channel(train_signal_clean)
        train_signal_clean = crutils.standardize_signal_per_channel(train_signal_clean, mean_train, std_train)
        if ds_type != "train":
            self.signal_clean = crutils.standardize_signal_per_channel(signal_clean, mean_train, std_train)
        else:
            self.signal_clean = train_signal_clean

        self.labels_bin = labels_bin
        self.labels_class = labels_class
        self.num_classes = self.labels_bin.shape[1]
        self.channels = self.signal_clean.shape[2]

        # Make sampler
        if apply_sampler:
            self.sampler, self.class_weights = crutils.balance_input(self.signal_clean, self.labels_bin)
        else:
            self.class_weights = None
            self.sampler = None

        if shuffle and apply_sampler:
            shuffle = False

        if shuffle:
            permutation = np.random.permutation(self.signal_clean.shape[0])
            self.signal_clean = self.signal_clean[permutation, :, :]
            self.labels_bin = self.labels_bin[permutation, :]

    def __len__(self):
        return self.signal_clean.shape[0]

    def __getitem__(self, idx):
        sample = torch.from_numpy(self.signal_clean[idx]).float()
        label = torch.from_numpy(self.labels_bin[idx]).float()
        return sample, label


def make_dataloader(ds: ECGDataset, batch_size: int, shuffle: bool = False):

    if ds.sampler is not None:
        shuffle = False

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        sampler=ds.sampler,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=True,
        num_workers=0,
    )
    return loader
