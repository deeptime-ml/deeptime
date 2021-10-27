from typing import Optional, List

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ...base import Transformer, Model, EstimatorTransformer
from ...base_torch import DLEstimatorMixin
from ...util.torch import map_data, MLP


class TAEModel(Model, Transformer):
    r""" Model produced by time-lagged autoencoders. Contains the encoder, decoder, and can transform data to
    the latent code.

    Parameters
    ----------
    encoder : torch.nn.Module
        The encoder module.
    decoder : torch.nn.Module
        The decoder module.
    device : torch device or None, default=None
        The device to use.
    dtype : numpy datatype, default=np.float32
        The dtype that is used for transformation of data.

    See Also
    --------
    TAE
    """

    def __init__(self, encoder, decoder, device=None, dtype=np.float32):
        self._encoder = encoder
        self._decoder = decoder
        self._device = device
        self._dtype = dtype

    @property
    def encoder(self):
        r""" The encoder.

        :type: torch.nn.Module
        """
        return self._encoder

    @property
    def decoder(self):
        r""" The decoder.

        :type: torch.nn.Module
        """
        return self._decoder

    def _encode(self, x: torch.Tensor):
        return self._encoder(x)

    def transform(self, data, **kwargs):
        r""" Transforms a trajectory (or a list of trajectories) by passing them through the encoder network.

        Parameters
        ----------
        data : array_like or list of array_like
            The trajectory data.
        **kwargs
            Ignored.

        Returns
        -------
        latent_code : ndarray or list of ndarray
            The trajectory / trajectories encoded to the latent representation.
        """
        out = []
        for data_tensor in map_data(data, device=self._device, dtype=self._dtype):
            out.append(self._encode(data_tensor).cpu().numpy())
        return out if len(out) > 1 else out[0]


class TAE(EstimatorTransformer, DLEstimatorMixin):
    r""" Time-lagged autoencoder. :footcite:`wehmeyer2018timelagged`

    Parameters
    ----------
    encoder : torch.nn.Module
        Encoder module.
    decoder : torch.nn.Module
        Decoder module, its input features should be compatible with the encoder's output features.
    optimizer : str or callable, default='Adam'
        The optimizer to use, defaults to Adam. If given as string, can be one of 'Adam', 'SGD', 'RMSProp'.
        In case of a callable, the callable should take a `params` parameter list and a `lr` learning rate, yielding
        an optimizer instance based on that.
    learning_rate : float, default=3e-4
        The learning rate that is used for the optimizer. Defaults to the Karpathy learning rate.

    References
    ----------
    .. footbibliography::
    """
    _MUTABLE_INPUT_DATA = True

    def __init__(self, encoder: nn.Module, decoder: nn.Module, optimizer='Adam', learning_rate=3e-4):
        super().__init__()
        self._encoder = encoder
        self._decoder = decoder
        self.learning_rate = learning_rate
        self.setup_optimizer(optimizer, list(encoder.parameters()) + list(decoder.parameters()))
        self._mse_loss = nn.MSELoss(reduction='sum')
        self._train_losses = []
        self._val_losses = []

    def evaluate_loss(self, x: torch.Tensor, y: torch.Tensor):
        r""" Evaluates the loss based on input tensors.

        Parameters
        ----------
        x : torch.Tensor
            The tensor that is passed through encoder and decoder networks.
        y : torch.Tensor
            The tensor the forward pass is compared against.

        Returns
        -------
        loss : torch.Tensor
            The loss.
        """
        return self._mse_loss(y, self._decoder(self._encoder(x)))

    @property
    def train_losses(self) -> np.ndarray:
        r""" The collected train scores. First dimension contains the step, second dimension the score. Initially empty.

        :type: (T, 2) ndarray
        """
        return np.array(self._train_losses)

    @property
    def validation_losses(self) -> np.ndarray:
        r""" The collected validation scores. First dimension contains the step, second dimension the score.
        Initially empty.

        :type: (T, 2) ndarray
        """
        return np.array(self._val_losses)

    def fit(self, data_loader: DataLoader, n_epochs: int = 5, validation_loader: Optional[DataLoader] = None, **kwargs):
        r""" Fits the encoder and decoder based on data. Note that a call to fit does not reset the weights in the
        networks that are currently in :attr:`encoder` and :attr:`decoder`.

        Parameters
        ----------
        data_loader : DataLoader
            Data loader which yields batches of instantaneous and time-lagged data.
        n_epochs : int, default=5
            Number of epochs to train for.
        validation_loader : DataLoader, optional, default=MNone
            Data loader which  yields batches of instantaneous and time-lagged data for validation purposes. Can be
            left None, in which case no validation is performed.
        **kwargs
            Ignored kw.

        Returns
        -------
        self : TAE
            Reference to self.
        """
        step = 0
        for epoch in range(n_epochs):

            self._encoder.train()
            self._decoder.train()
            for batch_0, batch_t in data_loader:
                batch_0 = batch_0.to(device=self.device)
                batch_t = batch_t.to(device=self.device)

                self.optimizer.zero_grad()
                loss_value = self.evaluate_loss(batch_0, batch_t)
                loss_value.backward()
                self.optimizer.step()

                self._train_losses.append((step, loss_value.item()))

                step += 1

            if validation_loader is not None:
                self._encoder.eval()
                self._decoder.eval()

                with torch.no_grad():
                    lval = []
                    for batch_0, batch_t in data_loader:
                        batch_0 = batch_0.to(device=self.device)
                        batch_t = batch_t.to(device=self.device)

                        loss_value = self.evaluate_loss(batch_0, batch_t).item()
                        lval.append(loss_value)
                    self._val_losses.append((step, np.mean(lval)))

        return self

    def fetch_model(self) -> TAEModel:
        r""" Yields a new instance of :class:`TAEModel`.

        .. warning::
            The model can be subject to side-effects in case :meth:`fit` is called multiple times, as no deep copy
            is performed of encoder and decoder networks.

        Returns
        -------
        model : TAEModel
            The model.
        """
        from copy import deepcopy
        return TAEModel(deepcopy(self._encoder), deepcopy(self._decoder), device=self.device, dtype=self.dtype)


def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu


class TVAEModel(TAEModel):
    r""" Model produced by the time-lagged variational autoencoder (:class:`TVAE`).
    When transforming data, the encoded mean and log-variance are reparametrized and yielded.

    See Also
    --------
    TAEModel
    """
    def _encode(self, x: torch.Tensor):
        return _reparameterize(*self.encoder(x))


class TVAEEncoder(MLP):
    r""" A kind of :class:`MLP`, which maps the output through not one but two transformations so that it projects
    onto mean and log-variance.

    Parameters
    ----------
    units : list of int
        The units of the network.
    nonlinearity : callable
        The nonlinearity to use. Callable must produce a `torch.nn.Module` which implements the nonlinear operation.
    """

    def __init__(self, units: List[int], nonlinearity=nn.ELU):
        super().__init__(units[:-1], nonlinearity=nonlinearity, initial_batchnorm=False,
                         output_nonlinearity=nonlinearity)
        lat_in = units[-2]
        lat_out = units[-1]
        self._to_mu = nn.Linear(lat_in, lat_out)
        self._to_logvar = nn.Linear(lat_in, lat_out)

    def forward(self, inputs):
        out = self._sequential(inputs)
        return self._to_mu(out), self._to_logvar(out)


class TVAE(TAE):
    r""" The time-lagged variational autoencoder. For a description of the arguments please see :class:`TAE`.
    The one additional argument is `beta`, which is the KLD-weight during optimization.

    See Also
    --------
    TAE
    TVAEModel
    """
    def __init__(self, encoder: nn.Module, decoder: nn.Module, optimizer='Adam', learning_rate: float = 5e-4,
                 beta: float = 1.):
        super().__init__(encoder, decoder, optimizer=optimizer, learning_rate=learning_rate)
        self._beta = beta

    def evaluate_loss(self, x: torch.Tensor, y: torch.Tensor):
        r""" Evaluates the reconstruction loss and latent regularization loss, returns the sum.

        Parameters
        ----------
        x : torch.nn.Tensor
            Input tensor to be passed through the network.
        y : torch.nn.Tensor
            Tensor which `x` is compared against.

        Returns
        -------
        loss : torch.nn.Tensor
            The loss.
        """
        mu, logvar = self._encoder(x)
        z = _reparameterize(mu, logvar)
        y_hat = self._decoder(z)
        mse_val = self._mse_loss(y_hat, y)
        kld = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)
        return mse_val + self._beta * kld / float(y.shape[1])

    def fetch_model(self) -> TVAEModel:
        r""" Yields a new instance of :class:`TVAEModel`.

        .. warning::
            The model can be subject to side-effects in case :meth:`fit` is called multiple times, as no deep copy
            is performed of encoder and decoder networks.

        Returns
        -------
        model : TVAEModel
            The model.
        """
        from copy import deepcopy
        return TVAEModel(deepcopy(self._encoder), deepcopy(self._decoder), self.device, self.dtype)
