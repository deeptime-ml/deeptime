from typing import Optional

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..base import Transformer, Model
from ..base_torch import DLEstimator
from ..util.pytorch import map_data


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

    def transform(self, data, **kwargs):
        out = []
        for data_tensor in map_data(data, device=self._device, dtype=self._dtype):
            out.append(self._encoder(data_tensor).cpu().numpy())
        return out if len(out) > 1 else out[0]


class TAE(DLEstimator, Transformer):
    _MUTABLE_INPUT_DATA = True

    def __init__(self, encoder: nn.Module, decoder: nn.Module, optimizer='Adam', learning_rate=5e-4):
        super().__init__()
        self._encoder = encoder
        self._decoder = decoder
        self.learning_rate = learning_rate
        self.setup_optimizer(optimizer, list(encoder.parameters()) + list(decoder.parameters()))
        self._loss = nn.MSELoss(size_average=False)
        self._train_losses = []
        self._val_losses = []

    def evaluate_loss(self, x: torch.Tensor, y: torch.Tensor):
        return self._loss(y, self._decoder(self._encoder(x)))

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

