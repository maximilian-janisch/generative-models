"""
Useful functions for data importing and processing.
"""

import pickle
import pandas as pd
import sklearn.base
import torch
from sklearn.model_selection import train_test_split
import numpy as np


# Data Processing
def process_toy_data(df, train_test_split_ratio=0.8, cols_to_drop=None, scaler=None,
                     labels_dim=0,
                     *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor, sklearn.base.TransformerMixin]:
    """
    Function to pre-process a dataset provided either as `pandas.DataFrame` or as `numpy.array`. Splits the dataset
     into a train and test set, and returns them in the form of `torch.Tensor`.

     For labeled data, the labels should be the last columns of the `pandas.DataFrame`.
    :param df: Dataset.
    :param train_test_split_ratio: Ratio with which to split the data into training and testing set. If equal to 1,
     the test set is empty. If equal to 0, the train set is empty.
    :param cols_to_drop: Columns to drop if passed as a `pandas.DataFrame`. Pass None if no columns should be dropped.
    :param scaler: Scaler with which to scale the data. The scaler will be fitted and transforms the data.
    :param labels_dim: Used in order to process labels in the dataset differently for conditional sampling.
    :param args, kwargs: Unused.
    :return: Tuple containing the train data as `torch.Tensor`, the test data as `torch.Tensor`,
     and also the scaler already passed to the function.
    """
    df = df.drop(columns=cols_to_drop, axis=1) if len(cols_to_drop) > 0 else df
    df = df.to_numpy() if isinstance(df, pd.DataFrame) else df

    # Scale the data
    if labels_dim == 0:
        df_no_labels = scaler.fit_transform(df) if scaler else df  # Possibility of not scaling
    else:
        df_no_labels = scaler.fit_transform(df[:, :-labels_dim]) if scaler else df  # Possibility of not scaling

    df = np.concatenate((df_no_labels, df[:, -labels_dim:]), axis=1) if labels_dim > 0 else df_no_labels

    # Split into train and test
    n_train_data, n_test_data = train_test_split(df, test_size=(1 - train_test_split_ratio))

    return n_train_data, n_test_data, scaler


# Get the processed data from a file
def get_data_from_file(data_file_path: str,
                       num_df: int,
                       cols_to_drop: list[str] = None,
                       scaler=None,
                       train_test_split_ratio=0.8,
                       label_dim=0, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor, sklearn.base.TransformerMixin]:
    """
    Helper function to read a dataset from a file located at `file_path`. This file should be a pickle containing
    either a `pandas.DataFrame` or a `np.array`, or a list of `pandas.DataFrame` and `np.array`. If the file contains a
    list, use the parameter num_df to select which dataset should be used.
    TODO: Make it so that if the file contains a list, then every dataset from that list is loaded.
    :param data_file_path: Path to file.
    :param num_df: See above.
    :param batch_size: Batch size for the train and test loaders.
    :param cols_to_drop: See `process_toy_data`.
    :param scaler: See `process_toy_data`.
    :param train_test_split_ratio: See `process_toy_data`.
    :param label_dim: See `process_toy_data`.
    :param args, kwargs: Passed on to `process_toy_data`.
    :return: See `process_toy_data`.
    """
    with open(data_file_path, "rb") as f:
        dynamic_dataset = pickle.load(f)[num_df] if num_df > -1 else pickle.load(f)

    return process_toy_data(dynamic_dataset,
                            train_test_split_ratio,
                            cols_to_drop,
                            scaler,
                            label_dim,
                            *args, **kwargs)
