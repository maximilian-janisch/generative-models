"""
Defines a helper function to instantiate a Multi-Layer Perceptron.
"""

import torch.nn as nn


def get_multilayer_perceptron(in_features=1,
                              out_features=1,
                              hidden_layer_input_sizes=(100, 100, 100),
                              dropouts=(0, 0, 0),
                              batch_normalization=False,
                              bias=True,
                              activation=nn.ReLU,
                              final_activation=nn.Sigmoid):
    """
    Returns a PyTorch Model using the torch.nn.Sequential constructor which is an MLP. Input to the model must be 1d of
    size in_features (i.e. what is fed into the model must be of shape batch_size x in_features).
    Output will be 1d of size output_size (i.e. of shape batch_size x out_features).

    :param in_features: Size of the input.
    :param out_features: Size of the output.
    :param hidden_layer_input_sizes: Input sizes of the hidden layers. Can be a tuple of arbitrary length.
    The last entry of this tuple is the input size of the output layer.
    :param dropouts: Dropouts to use after the linear layers. The first entry of this tuple will be the dropout used
    after the input layer. There is no dropout after the output layer, therefore, the length of the dropouts tuple must
    equal the length of the hidden_layer_sizes tuple.
    :param batch_normalization: If true, will add a torch.nn.BatchNormalization1d layer after the input layer and each
    hidden layer, but not the output layer.
    :param bias: Whether to use a bias for the linear layers (i.e. whether to use affine or linear functions).
    :param activation: Activation to use for the input and hidden layers (but not the output layer).
    :param final_activation: Activation to use after the output layer.
    """
    if batch_normalization:
        layers = [
            nn.Linear(in_features=in_features, out_features=hidden_layer_input_sizes[0], bias=bias),
            nn.BatchNorm1d(num_features=hidden_layer_input_sizes[0]),
            activation(),
            nn.Dropout(p=dropouts[0])
        ]
    else:
        layers = [
            nn.Linear(in_features=in_features, out_features=hidden_layer_input_sizes[0], bias=bias),
            activation(),
            nn.Dropout(p=dropouts[0])
        ]

    for i, layer_size in enumerate(hidden_layer_input_sizes[:-1]):
        layers.append(nn.Linear(in_features=layer_size, out_features=hidden_layer_input_sizes[i+1], bias=bias))
        if batch_normalization:
            layers.append(nn.BatchNorm1d(num_features=hidden_layer_input_sizes[i+1]))
        layers.append(activation())
        layers.append(nn.Dropout(p=dropouts[i+1]))

    layers.extend([
        nn.Linear(in_features=hidden_layer_input_sizes[-1],
                  out_features=out_features, bias=bias),
        final_activation()
    ])

    return nn.Sequential(*layers)
