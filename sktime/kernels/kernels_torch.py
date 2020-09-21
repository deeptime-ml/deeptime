from typing import Union

import numpy as np
import torch

from . import GaussianKernel

TensorOrArray = Union[np.ndarray, torch.Tensor]


class TorchGaussianKernel(GaussianKernel):

    @staticmethod
    def cdist(x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(
            x2_norm.t(),
            x1,
            x2.t(),
            alpha=-2
        ).add_(x1_norm).clamp_min_(1e-16)
        return res

    def apply_torch(self, data_1: torch.Tensor, data_2: torch.Tensor):
        #differences = torch.unsqueeze(data_1, -2) - torch.unsqueeze(data_2, -3)
        #distance_matrix = torch.sum(torch.pow(differences, 2), dim=-1)
        distance_matrix = TorchGaussianKernel.cdist(data_1, data_2)
        return torch.exp(-distance_matrix / (2. * self.sigma ** 2))

    def apply(self, data_1: TensorOrArray, data_2: TensorOrArray) -> TensorOrArray:
        if isinstance(data_1, np.ndarray) and isinstance(data_2, np.ndarray):
            return super().apply(data_1, data_2)
        elif isinstance(data_1, torch.Tensor) and isinstance(data_2, torch.Tensor):
            return self.apply_torch(data_1, data_2)
        else:
            raise ValueError(f"Arguments can only be both numpy arrays or torch tensors but were {data_1} and {data_2}")
