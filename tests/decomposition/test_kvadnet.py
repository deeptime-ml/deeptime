import pytest

pytest.importorskip("torch")

import numpy as np
import torch

import sktime
from sktime.decomposition.kvad import kvad
import sktime.decomposition.kvadnet as kvadnet


def test_whiten():
    chi_X = np.random.uniform(-1, 1, size=(1000, 50))
    with torch.no_grad():
        xw = kvadnet.whiten(torch.from_numpy(chi_X), mode='clamp', epsilon=1e-10).numpy()
    np.testing.assert_array_almost_equal(xw.mean(axis=0), np.zeros((xw.shape[1],)))
    cov = 1 / (xw.shape[0] - 1) * xw.T @ xw
    np.testing.assert_array_almost_equal(cov, np.eye(50))


def test_score():
    dataset = sktime.data.bickley_jet(n_particles=100, n_jobs=1)
    ds_2d = dataset.endpoints_dataset()
    ds_3d = ds_2d.to_3d().cluster(4)
    kvad_3d_cluster_model = kvad(ds_3d.data, ds_3d.data_lagged, Y=ds_2d.data_lagged)
    with torch.no_grad():
        chi_X = torch.from_numpy(ds_3d.data)
        Y = torch.from_numpy(ds_2d.data_lagged)
        score_net = kvadnet.kvad_score(chi_X, Y, mode='clamp', epsilon=1e-10).numpy()
    np.testing.assert_almost_equal(score_net, kvad_3d_cluster_model.score, decimal=1)
