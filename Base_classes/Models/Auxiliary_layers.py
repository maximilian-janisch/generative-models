"""
File which defines auxiliary layers such as SplitLayer, ConcatLayer, and ReparametrizationTrick using the
pytorch_symbolic LambdaOpLayer.
"""

import torch
from pytorch_symbolic.useful_layers import LambdaOpLayer


# Symbolic layer for splitting the data according to specified dimensions
def SplitLayer(split_list):
    return LambdaOpLayer(lambda data: torch.split(data, split_list, 1))


# Symbolic layer for concatenating two tensors into one
def ConcatLayer():
    return LambdaOpLayer(lambda *inputs: torch.cat(inputs, dim=1))


# Reparameterization trick for Variational Auto-Encoders
def ReparameterizationTrick():
    def reparameterize(mu_logvar):
        length = mu_logvar.shape[1]//2
        mu_logvar_list = torch.split(mu_logvar, [length, length], 1)
        mu = mu_logvar_list[0]
        logvar = mu_logvar_list[1]
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    return LambdaOpLayer(lambda mu_logvar: reparameterize(mu_logvar))
