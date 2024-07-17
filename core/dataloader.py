# Loading the Physionet data

import ast
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import wfdb  # process physiological signals


def load_raw_data(df: pd.DataFrame, data_path: Path, use_low_rate: bool = True) -> np.ndarray:
    """Loads the raw data as suggested by physionet.

    Args:
        df (pd.DataFrame): DataFrame with the annotations
        data_path (Path): Path to the data
        use_low_rate (bool): If true, use the sampling rate 100. Otherwise 500.

    Returns:
        np.ndarray: Numpy array with the raw data
    """

    if use_low_rate:
        data = [wfdb.rdsamp(data_path.joinpath(f)) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(data_path.joinpath(f)) for f in df.filename_hr]
    data = np.array([signal for signal, _ in data])
    return data


def data_split(raw_data: np.ndarray, annotations: pd.DataFrame) -> Dict:
    """ Splits the data as suggested by physionet.

    Args:
        raw_data (np.ndarray): Array with the raw data
        annotations (pd.DataFrame): DataFrame with the annotations

    Returns:
        Dict: Dict with the data splitted
    """

    # split data into train, val and test
    test_fold = 10
    val_fold = 9
    train_fold = 8

    # train
    data_train = raw_data[np.where(annotations.strat_fold <= train_fold)]
    label_train = annotations[(annotations.strat_fold <= train_fold)].diagnostic_superclass
    annot_train = annotations[(annotations.strat_fold <= train_fold)]

    # validation
    data_val = raw_data[np.where(annotations.strat_fold == val_fold)]
    label_val = annotations[annotations.strat_fold == val_fold].diagnostic_superclass
    annot_val = annotations[(annotations.strat_fold <= val_fold)]

    # test
    data_test = raw_data[np.where(annotations.strat_fold == test_fold)]
    label_test = annotations[annotations.strat_fold == test_fold].diagnostic_superclass
    annot_test = annotations[(annotations.strat_fold <= test_fold)]

    output = {
        "train": {
            "data": data_train,
            "labels": label_train,
            "annotations": annot_train
        },
        "val": {
            "data": data_val,
            "labels": label_val,
            "annotations": annot_val
        },
        "test": {
            "data": data_test,
            "labels": label_test,
            "annotations": annot_test
        }
    }

    return output


def load_data(data_path: Path = Path('physionet.org/files/ptb-xl/1.0.2'), sampling_rate: int = 100) -> Dict:
    """ Loads the data and applies the workflow suggested by physionet.
        By default we use the version with a sampling rate of 100Hz.

    Args:
        data_path (Path, optional): Path to the data.
                                    Defaults to Path('physionet.org/files/ptb-xl/1.0.2').
        sampling_rate (int, optional): Sampling rate. Defaults to 100.

    Returns:
        Dict: Dict with the data
    """

    data_path = Path(data_path) if not isinstance(data_path, Path) else data_path

    if not data_path.exists():
        raise FileNotFoundError(f"The path {data_path} doesn't exist.")

    annot_path = data_path.joinpath("ptbxl_database.csv")
    if not annot_path.exists():
        raise FileNotFoundError(f"The path {annot_path} doesn't exist.")

    diagnostic_path = data_path.joinpath("scp_statements.csv")
    if not diagnostic_path.exists():
        raise FileNotFoundError(f"The path {diagnostic_path} doesn't exist.")

    # load and convert annotation data
    annotations = pd.read_csv(annot_path, index_col='ecg_id')
    annotations.scp_codes = annotations.scp_codes.apply(lambda x: ast.literal_eval(x))

    # load raw signal data
    raw_data = load_raw_data(annotations, data_path)

    # load scp_statements.csv for diagnostic aggregation
    agg_df = pd.read_csv(diagnostic_path, index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    def aggregate_diagnostic(y_dic: Dict) -> List:
        tmp = [agg_df.loc[key].diagnostic_class for key in y_dic.keys() if key in agg_df.index]
        return list(set(tmp))

    # apply diagnostic superclass
    annotations['diagnostic_superclass'] = annotations.scp_codes.apply(aggregate_diagnostic)

    channels_name = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    sz_raw = len(raw_data)
    if len(annotations) != sz_raw:
        raise Exception("The number of annotations and data doesn't match.")

    sz_raw_sample = len(raw_data[0])
    for i in range(1, sz_raw):
        if sz_raw_sample != len(raw_data[i]):
            raise Exception("The data doesn't have consistent size.")

    annotations.sex = annotations.sex.map({0: 'male', 1: 'female'})
    output = data_split(raw_data, annotations)

    output["channels_name"] = channels_name
    output["sampling_rate"] = sampling_rate

    return output


def get_patient_id_ecg_ids(patient_id: int, annotations: pd.DataFrame) -> List:
    """ Get all the ECGs from a target patient.

    Args:
        patient_id (int): target patient id
        annotations (pd.DataFrame): annotations of the ECGs

    Returns:
        List: List with all the ECG ids for the patient and respective dates
    """

    filtered_df = annotations[annotations['patient_id'] == patient_id]
    if len(filtered_df) == 0:
        raise Exception(f"The patient {patient_id} has no records.")

    filtered_df = filtered_df.sort_values(by='recording_date')

    ecg_ids_with_dates = list(zip(filtered_df.index.tolist(), filtered_df['recording_date'].tolist()))

    return ecg_ids_with_dates


def get_annotations_from_ecg_id(ecg_id: int, annotations: pd.DataFrame) -> pd.DataFrame:
    """ Get the annotations from a target ECG.

        Args:
            ecg_id (int): target ecg id
            annotations (pd.DataFrame): annotations of the ECGs

        Returns:
            pd.DataFrame: annotations of the ECG
        """

    try:
        annot = annotations.loc[ecg_id]
        return annot
    except KeyError:
        raise Exception(f"The ECG ID {ecg_id} has no annotations.")


def get_signal_from_ecg_id(ecg_id: int, raw_data: np.ndarray, channel: int = -1) -> np.ndarray:
    """ Get the annotations from a target ECG.

        Args:
            ecg_id (int): target ecg id
            raw_data (pd.DataFrame): raw data of the ECGs
            channel (int, optional): required channel, by default all (-1)

        Returns:
            np.ndarray: Signal requested
        """

    n_channels = raw_data.shape[2]
    if channel >= n_channels and channel != -1:
        raise IndexError(f"The raw data has {n_channels} channels. Channel {channel} was requested.")

    try:
        annot = raw_data[ecg_id, :, channel] if channel != -1 else raw_data[ecg_id, :, :]
        return annot
    except KeyError:
        raise Exception(f"The ECG ID {ecg_id} has no annotations.")
