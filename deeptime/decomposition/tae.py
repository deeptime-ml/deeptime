from typing import Optional, List

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..base import Transformer, Model
from ..base_torch import DLEstimator
from ..util.torch import map_data, MLP


class TAEModel(Model, Transformer):

    def __init__(self, encoder, decoder, device=None, dtype=np.float32):
        self._encoder = encoder
        self._decoder = decoder
        self._device = device
        self._dtype = dtype

    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder

    def _encode(self, x: torch.Tensor):
        return self._encoder(x)

    def transform(self, data, **kwargs):
        out = []
        for data_tensor in map_data(data, device=self._device, dtype=self._dtype):
            out.append(self._encode(data_tensor).cpu().numpy())
        return out if len(out) > 1 else out[0]


class TAE(DLEstimator, Transformer):
    _MUTABLE_INPUT_DATA = True

    def __init__(self, encoder: nn.Module, decoder: nn.Module, optimizer='Adam', learning_rate=5e-4):
        super().__init__()
        self._encoder = encoder
        self._decoder = decoder
        self.learning_rate = learning_rate
        self.setup_optimizer(optimizer, list(encoder.parameters()) + list(decoder.parameters()))
        self._mse_loss = nn.MSELoss(reduction='sum')
        self._train_losses = []
        self._val_losses = []

    def evaluate_loss(self, x: torch.Tensor, y: torch.Tensor):
        return self._mse_loss(y, self._decoder(self._encoder(x)))

    def fit(self, data_loader: DataLoader, n_epochs: int = 5, validation_loader: Optional[DataLoader] = None, **kwargs):
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

                self._train_losses.append((epoch, loss_value.item()))

            if validation_loader is not None:
                self._encoder.eval()
                self._decoder.eval()

                with torch.no_grad():
                    for batch_0, batch_t in data_loader:
                        batch_0 = batch_0.to(device=self.device)
                        batch_t = batch_t.to(device=self.device)

                        loss_value = self.evaluate_loss(batch_0, batch_t)
                        self._val_losses.append((epoch, loss_value.item()))

        return self

    def fetch_model(self) -> TAEModel:
        return TAEModel(self._encoder, self._decoder, device=self.device, dtype=self.dtype)

    def transform(self, data, **kwargs):
        return self.fetch_model().transform(data)


def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu


class TVAEModel(TAEModel):

    def _encode(self, x: torch.Tensor):
        return _reparameterize(*self.encoder(x))


class TVAEEncoder(MLP):

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
    def __init__(self, encoder: nn.Module, decoder: nn.Module, optimizer='Adam', learning_rate: float = 5e-4,
                 beta: float = 1.):
        super().__init__(encoder, decoder, optimizer=optimizer, learning_rate=learning_rate)
        self._beta = beta

    def evaluate_loss(self, x: torch.Tensor, y: torch.Tensor):
        mu, logvar = self._encoder(x)
        z = _reparameterize(mu, logvar)
        y_hat = self._decoder(z)
        mse_val = self._mse_loss(y_hat, y)
        kld = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)
        return mse_val + self._beta * kld / float(y.shape[1])

    def fetch_model(self) -> TAEModel:
        return TVAEModel(self._encoder, self._decoder, self.device, self.dtype)
