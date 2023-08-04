"""
Base class from which all generative models inherit.
"""


from abc import ABC, abstractmethod

import torch
import torch.utils.data
import torch.nn as nn
import torch.optim.optimizer


class GenerativeModel(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes the model using the hyperparameters *args, **kwargs.

        The only hyperparameters passed to the initializer of this class should be those that are specific to the
        architecture of the model. These may include for example number of hidden layers for the MLPs in a VAE, or
        whether the decoder of the VAE is stochastic.

        The `self.sampler` is used in order to resample from the learned distribution, while each of the `self.losses`
        is a module that takes training data as input and returns a loss corresponding to the performance of that module
        on the training data.
        """
        self.sampler = self.losses = None

    @abstractmethod
    def __call__(self,
                 n_samples: int,
                 labels=None,
                 *args,
                 **kwargs) -> torch.Tensor:
        """
         Either pass an integer or a torch tensor as argument `n_samples_or_labels`. If an integer is passed, the function
        resamples `n_samples` from the unconditional distribution learned by the CVAE. If a `torch.Tensor` is passed, resample
        `n_samples_or_labels.size()[0]` samples, conditionally on each label passed as argument. The tensor must be of
        shape (`n_samples`, `labels_dim`).

        Output will be a tensor of the shape (`n_samples`, `data_dim`).
        """
        pass


    def fit_one_epoch_from_dataloader(self,
                                      train_loader,
                                      verbose=True,
                                      validation_loader: torch.utils.data.DataLoader = None,
                                      *args, **kwargs) -> None:
        """
        Perform one epoch of training using a given PyTorch dataloader. For explanation of the arguments, see the
        `fit_from_dataloader` method.

        The data is composed by the inputs and the labels.
         self.losses is a collection that contains "losses" objects, which are nn.Modules with a forward, an optimizer,
         and a float containing the cumulated loss during training.
        """
        for current_loss in self.losses:
            current_loss.cumulated_loss = 0

        for data in train_loader:
            for current_loss in self.losses:
                current_loss.optimizer.zero_grad()
                loss = current_loss(torch.Tensor(data).float())
                loss.backward()
                current_loss.cumulated_loss += loss.item()
                current_loss.optimizer.step()

    def fit_from_dataloader(self,
                            train_loader: torch.utils.data.DataLoader,
                            pre_initialize_weights=False,
                            epochs=1,
                            verbose=True,
                            validation_loader: torch.utils.data.DataLoader = None,
                            random_state=None,
                            *args, **kwargs) -> None:
        """
        Train the generative model, given a PyTorch dataloader.
        :param train_loader: An instance of a `torch.utils.data.DataLoader` (DataLoader).
        :param pre_initialize_weights: If True, use the `initialize` method in order to initialize weights. If False,
         let PyTorch initialize.
        :param epochs: Number of epochs to train.
        :param verbose: Whether to print the train loss after every epoch.
        :param validation_loader: DataLoader for validation data.
        :param random_state: Random state for the initialization of the weights.
        :param args, kwargs: Passed on.
        """
        try:  # check for optimizer
            assert all(loss.optimizer is not None for loss in self.losses)
        except (AttributeError, AssertionError):
            raise Exception("Please initialize the optimizer first using the set_optimizer method.")

        # Initialization
        for loss in self.losses:
            loss.train()

        if pre_initialize_weights:
            self.initialize(random_state)

        for ep in range(epochs):
            self.fit_one_epoch_from_dataloader(
                train_loader,
                verbose=verbose,
                validation_loader=validation_loader,
                *args, **kwargs
            )
            if verbose:  # todo: validation loss
                print('====> Epoch: {}; Cumulated training losses over epoch:'.format(ep),
                      [loss.cumulated_loss for loss in self.losses])

    def fit(self,
            X,
            batch_size=32,
            epochs=1,
            verbose=True,
            validation_set=None,
            validation_batch_size=32,
            *args, **kwargs) -> None:
        """
        Convenience function which wraps around the `fit_from_dataloader` method. Instead of passing a
        `torch.utils.data.DataLoader` for train and validation sets, you can pass them as `torch.tensor` or `np.array`.

        Shape of the arrays must be of the form (n_samples, ...).
        """
        self.fit_from_dataloader(
            train_loader=torch.utils.data.DataLoader(X, batch_size=batch_size),
            epochs=epochs,
            verbose=verbose,
            validation_loader=torch.utils.data.DataLoader(validation_set,
                                                          batch_size=validation_batch_size) if validation_set is not None else None,
            *args, **kwargs
        )

    def reconstruct(self, X: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Function to reconstruct a given input. Should only be overwritten by models which have the ability to do so,
        such as Auto-Encoders. For Generative models which cannot reconstruct the given input, this function will
        instead simply give a sample from the learned distribution, such that the sample has the same size as the
        input X.
        :param X: Input to reconstruct
        :param args, kwargs: Passed on.
        :return: Reconstruction of X or resampling of the same shape.
        """
        return self.__call__(X.shape[0], *args, **kwargs)

    def set_optimizer(self, optimizers: list[torch.optim.Optimizer] = torch.optim.Adam, *args, **kwargs):
        """
        Sets the optimizers for each of the losses of the model. Either pass one `torch.optim.optimizer` in order to
        set the same optimizer for all losses, or pass a list of `torch.optim.optimizer` to define different optimizers
        for each loss.

        The function makes reference to the `define_optimizer` method, see the documentation of that method for what options there are
        to define optimizers.
        """
        if isinstance(optimizers, list):
            for loss, optimizer in zip(self.losses, optimizers):
                loss.optimizer = self.define_optimizer(loss.parameters(), optimizer, **kwargs)
        else:
            for loss in self.losses:
                loss.optimizer = self.define_optimizer(loss.parameters(), optimizers, **kwargs)

    def initialize(self, random_state=None) -> None:
        """
        Randomizes the weights according to the method described in
        "Understanding the difficulty of training deep feedforward neural networks" - Glorot, X. & Bengio, Y. (2010).
        """
        if random_state is not None:
            torch.manual_seed(random_state)
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.05)

        for loss in self.losses:
            loss.apply(init_weights)

    @staticmethod
    def define_optimizer(parameters, optimizer, **kwargs) -> torch.optim.Optimizer:
        """
        Utility function that takes as input a class which inherits from `torch.optim.Optimizer`, as well as the
        hyperparameters for the optimizer, and returns an instance of `torch.optim.Optimizer`.

        The parameters over which the optimizer optimizes are given as the first argument.
        """
        return optimizer(parameters, **kwargs)
