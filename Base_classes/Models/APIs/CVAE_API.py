"""
Defines the API for a Conditional Variational Auto-Encoder (CVAE).

The class CVAE_API inherits from Base_classes.Generative_model.GenerativeModel and implements the abstract methods
defined there, fine-tuned for a CVAE, notably for training and sampling.
"""

from typing import Union

import numpy as np

import torch
import torch.nn as nn

from Base_classes.Models.Generative_model import GenerativeModel

from pytorch_symbolic import Input, SymbolicModel, optimize_module_calls
from pytorch_symbolic.useful_layers import ConcatLayer, LambdaOpLayer
from Base_classes.Models.Auxiliary_layers import ReparameterizationTrick, SplitLayer


class CVAE_API(GenerativeModel):
    def __init__(self,
                 data_dim,
                 label_dim,
                 latent_dim,
                 encoder,
                 decoder,
                 use_stochastic_decoder=True,
                 beta=1,
                 gamma=0.01,
                 *args, **kwargs):
        """
        Initializes the CVAE
        :param data_dim: Dimension of the input data (and thus also dimension of the reconstructed data).
        :param label_dim: Dimension of the labels (used for conditional sampling, set it =0 for unconditional sampling).
        :param latent_dim: Dimension of the latent space.
        :param encoder: PyTorch Module that is the encoder K_e used for the CVAE. Should take data of dimension data_dim
         as input and return data of dimension latent_dim.
        :param decoder: PyTorch Module that is the decoder K_d used for the CVAE. Should take data of dimension
         latent_dim as input and return either
          * if use_stochastic_decoder is True, a vector of dimension 2*data_dim, containing the mean and log variance of
            the Gaussian distribution from which the output is sampled ;
          * if use_stochastic_decoder is False, a vector of dimension data_dim.
        :param use_stochastic_decoder: If True, the decoder computes the mean and variance of a Gaussian distribution,
         from which we sample the reconstruction. If False, the decoder computes the reconstruction directly.
        :param beta, gamma: The total loss is pseudo_ce + beta * kld + gamma / kld. A "true" VAE has beta=1 and gamma=0.
         However, choosing a small gamma, such as gamma=0.01, may increase training stability.
        :param args, kwargs: Unused.
        """
        # declare the symbolic forward pass of the main module
        raw_data = Input(shape=(data_dim + label_dim,))
        data, label = SplitLayer([data_dim, label_dim])(raw_data)
        latent_raw = encoder(raw_data)  # Contains the concatenation of mean and log-variance
        latent = ReparameterizationTrick()(latent_raw)
        reconstructed_data = decoder(ConcatLayer(dim=1)(latent, label))

        # Losses
        kld = LambdaOpLayer(self.loss_KLD)(latent_raw)
        if use_stochastic_decoder:
            mse = LambdaOpLayer(self.pseudo_CE_stochastic_decoder)(data, reconstructed_data)
            reconstructed_data = ReparameterizationTrick()(reconstructed_data)
        else:
            mse = nn.MSELoss(data, reconstructed_data)

        total_loss = mse + beta * kld + gamma / kld

        # Define the model by setting its inputs and outputs
        self.losses = []
        self.losses.append(SymbolicModel(raw_data, total_loss))

        self.sampler = SymbolicModel([latent, label], reconstructed_data)

        # Keep track of the parameters of the model
        self.data_dim = data_dim
        self.label_dim = label_dim
        self.latent_dim = latent_dim
        self.use_stochastic_decoder = use_stochastic_decoder
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.gamma = gamma

        # Optimization for better performances (black magic, optional)
        optimize_module_calls()

    @staticmethod
    def loss_KLD(mu_logvar) -> torch.Tensor:
        """
        Given a pair of data_dim dimensional vectors mu and logvar, in the form of a batch_size x (2 * data_dim)-shaped
        torch.Tensor, this function computes the Kullback--Leibler divergence between a Gaussian distribution
        with mean mu and covariance matrix equal to the diagonal matrix whose entries are the element-wise exponential
        of logvar, and a standard normal distribution. This is then summed over the batch.

        The output is a torch.Tensor of shape (1, ).
        """
        dim = mu_logvar.shape[1]
        assert dim % 2 == 0
        mu, logvar = torch.split(mu_logvar, [dim // 2, dim // 2], 1)
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    @staticmethod
    def pseudo_CE_stochastic_decoder(input_, mu_logvar) -> torch.Tensor:
        """
        Computes the pseudo Cross-Entropy for the Gaussian CVAE. In the descriptions we will use the notation of an
        accompanying TeX document, however, we will omit the ^\theta and ^\phi. See also loss_KLD above.
        NOTE: For each fixed v, we only sample ONE Z_l, i.e. L=1.

        Note: This loss is for regression tasks, not for classification tasks.

        TODO: Implement stochastic decoder loss for classification tasks.

        :param input_: The original input v as (batch_size, data_dim)-shaped torch.Tensor.
        :param mu_logvar: Mu and logvar output of the decoder, as a (batch_size, (2 * data_dim))-shaped torch.Tensor.
        :returns: Pseudo CE as defined in (3.3) of accompanying TeX document.
        """
        # Reconstruction loss
        dim = mu_logvar.shape[1]
        assert dim % 2 == 0
        mu, logvar = torch.split(mu_logvar, [dim // 2, dim // 2], 1)
        pseudo_CE = torch.sum(
            torch.div((input_ - mu).pow(2), logvar.exp()))  # Note: Different for Bernoulli CVAEs.
        pseudo_CE += torch.sum(logvar) / 2
        return pseudo_CE

    def __call__(self,
                 n_samples_or_labels: Union[int, torch.Tensor],
                 custom_z_samples=None,
                 *args,
                 **kwargs):
        """
        Either pass an integer or a torch tensor as argument `n_samples_or_labels`. If an integer is passed, the function
        resamples `n_samples` from the unconditional distribution learned by the CVAE. If a `torch.Tensor` is passed, resample
        `n_samples_or_labels.size()[0]` samples, conditionally on each label passed as argument. The tensor must be of
        shape (`n_samples`, `labels_dim`).

        Output will be a tensor of the shape (`n_samples`, `data_dim`).

        `args` and `kwargs` are passed on.

        TODO: Sample from the unconditional distribution, even if we trained with labels
        => We need to learn the distribution of the labels.
        """
        if isinstance(n_samples_or_labels, torch.Tensor):
            z_samples = custom_z_samples if custom_z_samples is not None else np.random.normal(0, 1, (
                n_samples_or_labels.size()[0], self.latent_dim)
            )

            return self.sampler(torch.Tensor(z_samples).float(), n_samples_or_labels.float())
        else:
            if self.label_dim:
                raise NotImplementedError("Cannot do unconditional sampling if training was performed with labels.")

            z_samples = custom_z_samples if custom_z_samples is not None else np.random.normal(0, 1, (
                n_samples_or_labels, self.latent_dim)
            )

            return self.sampler(torch.Tensor(z_samples).float(), torch.Tensor())

    def reconstruct(self, X: torch.Tensor, labels=None, *args, **kwargs) -> torch.Tensor:
        """
        Function to reconstruct a given input. Should only be overwritten by models which have the ability to do so,
        such as Auto-Encoders. For Generative models which cannot reconstruct the given input, this function will
        instead simply give a sample from the learned distribution, such that the sample has the same size as the
        input X.
        :param X: Input to reconstruct, passed as `torch.Tensor`.
        :param labels: Passed on to the resampler.
        :param args, kwargs: Passed on.
        :return: Reconstruction of X or resampling of the same shape.
        """
        X_encoded = ReparameterizationTrick()(self.encoder(X))

        return self.__call__(X.shape[0], labels=labels, custom_z_samples=X_encoded)
