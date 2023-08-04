"""
Example configuration file for a Conditional Variational Autoencoder.
"""

import os

import torch
from torch import nn

from sklearn.preprocessing import StandardScaler

from Base_classes.Models.Multilayer_Perceptron import get_multilayer_perceptron


__all__ = {"config", "config_resampling"}

config = {
    "model_name": "CVAE",   # Unused
    "inference": True,
    "data_file_path": os.path.join("Datasets", "two_mixed_gaussians_labeled.pickle"),
    "train_test_split_ratio": 0.8,
    "cols_to_drop": [],
    "scaler": StandardScaler(),
    "num_df": -1,
    "file_name": os.path.join("Results", "Generated_data", "CVAE_for_two_mixed_gaussians_labeled.pickle"),
    "data_dim": 2,
    "latent_dim": 10,
    "label_dim": 1,
    "batch_size": 32,
    "epochs": 20,
    "loss_func": nn.MSELoss(),
    "lr": 0.001,
    "activation_function_in": nn.ReLU,
    "encoder_layers": [30, 40, 25],
    "decoder_layers": [25, 50, 40],
    "beta": 1,
    "optimizer": {"optimizers": torch.optim.Adam}
}

config["encoder_sub"] = {
        "in_features": config['data_dim']+config['label_dim'],
        'out_features': 2*config['latent_dim'],
        'hidden_layer_input_sizes': config['encoder_layers'],
        'activation': config['activation_function_in'],
        'final_activation': nn.Identity
}

config["decoder_sub"] = {
        'in_features': config['latent_dim']+config['label_dim'],
        'out_features': 2*config['data_dim'],
        'hidden_layer_input_sizes': config['decoder_layers'],
        'activation': config['activation_function_in'],
        'final_activation': nn.Identity
}

config.update({
    "encoder": get_multilayer_perceptron(**config["encoder_sub"]),
    "decoder": get_multilayer_perceptron(**config["decoder_sub"])
})


config_resampling = {
    "n_samples": 5000,
    "labels": [1, -1],
    "plot_titles": ["Generated results with label=1", "Generated results with label=-1"],
    "graph_name": os.path.join("Results", "Generated_data", "CVAE_for_two_mixed_gaussians_labeled.png"),
}