"""
Example file

Instantiate, train, and evaluate a model.
"""
import time
import torch
import numpy as np

from Visualization import plot_data_and_reconstruction
from Base_classes.Data_processing.helpers import get_data_from_file
from Base_classes.Models.APIs.CVAE_API import CVAE_API
from Config_files.Inference.CVAE_for_mixture_gaussians import config, config_resampling

if __name__ == '__main__':
    if config["inference"]:  # load a given model and train, test it. Alternatively, one can do hyperparameter optimization (to be implemented)
        print("Processing the data...")
        train_data, test_data, scaler = get_data_from_file(**config)  # load data according to rules specified in config

        print("Getting the model...")
        model = CVAE_API(**config)  # instantiation of the model
        model.set_optimizer(**config["optimizer"])

        print("Training the model...")
        current_time = time.time()
        model.fit(train_data, **config)  # training of the model
        print("Training time: ", time.time() - current_time)

        # we now use the Conditional Variational Auto-Encoder to resample conditionally on all the labels specified in config_resampling
        original_dataset = test_data
        resampled_datasets = []
        for label in config_resampling["labels"]:
            resampled_datasets.append(
                model(torch.full(
                    size=(original_dataset.shape[0], 1),
                    fill_value=label
                ))
            )
        resampled_datasets = [dataset.detach().numpy() for dataset in resampled_datasets]

        # Plotting of the results
        total_dataset = np.concatenate(resampled_datasets)
        try:
            plot_titles = ["Original data"] + config_resampling["plot_titles"] + ["Data across all labels"]
        except KeyError:
            plot_titles = None

        fig = plot_data_and_reconstruction([scaler.inverse_transform(dataset)
                                            for dataset in [original_dataset[:, :-config["label_dim"]]] + resampled_datasets + [total_dataset]],
                                           plot_titles)
        fig.show()
