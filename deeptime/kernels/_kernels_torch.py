from typing import Union

import numpy as np
import torch

from . import GaussianKernel

TensorOrArray = Union[np.ndarray, torch.Tensor]


class TorchGaussianKernel(GaussianKernel):
    r""" Gaussian kernel which has a dependency to PyTorch and can handle evaluation of Gram matrices as well as
    (over the batch-size) vectorized evaluations of the kernel.

    See Also
    --------
    GaussianKernel
    """

    def __init__(self, sigma):
        np_sigma = sigma
        if isinstance(np_sigma, torch.Tensor):
            np_sigma = sigma.detach().cpu().numpy()
        super().__init__(np_sigma)
        self._sigma = sigma
        self.impl = 'cdist'

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
        distance_matrix = TorchGaussianKernel.cdist(data_1, data_2)
        return torch.exp(-distance_matrix / (2. * self.sigma ** 2))

    def _evaluate(self, x, y) -> TensorOrArray:
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            dxy_squared = torch.pow(x - y, 2).sum(-1).clamp_min_(0.)
            return torch.exp(-dxy_squared / (2. * self.sigma ** 2))
        else:
            out = []
            for batch_ix in range(len(x)):
                out.append(super()._evaluate(x[batch_ix], y[batch_ix]))
            return np.array(out)

    def apply(self, data_1: TensorOrArray, data_2: TensorOrArray) -> TensorOrArray:
        if isinstance(data_1, np.ndarray) and isinstance(data_2, np.ndarray):
            return super().apply(data_1, data_2)
        elif isinstance(data_1, torch.Tensor) and isinstance(data_2, torch.Tensor):
            return self.apply_torch(data_1, data_2)
        else:
            raise ValueError(f"Arguments can only be both numpy arrays or torch tensors but were {data_1} and {data_2}")
