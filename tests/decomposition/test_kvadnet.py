import pytest

from deeptime.kernels import GaussianKernel

pytest.importorskip("torch")

from deeptime.data import bickley_jet
from deeptime.decomposition import KVAD
from deeptime.decomposition.deep._kvadnet import whiten, kvad_score

import numpy as np
import torch


def test_whiten():
    chi_X = np.random.uniform(-1, 1, size=(1000, 50))
    with torch.no_grad():
        xw = whiten(torch.from_numpy(chi_X), mode='clamp', epsilon=1e-10).numpy()
    np.testing.assert_array_almost_equal(xw.mean(axis=0), np.zeros((xw.shape[1],)))
    cov = 1 / (xw.shape[0] - 1) * xw.T @ xw
    np.testing.assert_array_almost_equal(cov, np.eye(50))


def test_score():
    kernel = GaussianKernel(1.)
    dataset = bickley_jet(n_particles=100, n_jobs=1)
    ds_2d = dataset.endpoints_dataset()
    ds_3d = ds_2d.to_3d().cluster(4)
    ref_score = KVAD(kernel, epsilon=1e-6).fit_fetch(ds_3d).score
    with torch.no_grad():
        chi_x = torch.from_numpy(ds_3d.data)
        y = torch.from_numpy(ds_3d.data_lagged)
        score_net = kvad_score(chi_x, y, kernel=kernel).numpy()
    np.testing.assert_almost_equal(score_net, ref_score, decimal=1)
