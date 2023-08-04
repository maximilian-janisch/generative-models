import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


def plot_data_and_reconstruction(list_of_datasets: list[np.array], plot_titles: list[str] = None) -> plt.figure:
    """
    Plot multiple datasets, passed as a list of numpy arrays. All numpy arrays must be of shape (n_samples, dimension).
    If the dimension is strictly greater than 2, PCA is used to project the data into 2 dimensional space.

    The datasets are plotted using Python subplots. One subplot is created for each dataset.

    You can pass a list for the titles of each subplot as the `list_of_titles` argument.
    """
    fig, axs = plt.subplots(1, len(list_of_datasets), figsize=(len(list_of_datasets) * 5, 5))

    def do_PCA(data):
        if isinstance(data, torch.Tensor):
            data = data.detach().numpy()
        elif isinstance(data, pd.DataFrame):
            data = data.to_numpy()
        return PCA(n_components=2).fit_transform(data) if data.shape[1] > 2 else data

    list_of_datasets = [do_PCA(dataset) for dataset in list_of_datasets]

    for ax, dataset in zip(axs, list_of_datasets):
        ax.scatter(dataset[:, 0], dataset[:, 1])

    if plot_titles is not None:
        for ax, title in zip(axs, plot_titles):
            ax.set_title(title)

    return fig
