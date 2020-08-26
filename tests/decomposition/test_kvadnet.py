import numpy as np
import pytest
import torch

import sktime
import sktime.decomposition.kvad as kvad
import sktime.decomposition.kvadnet as kvadnet


@pytest.mark.parametrize('sigma', [1., 5., 10.])
def test_gramian(sigma):
    with torch.no_grad():
        sigma = 1.
        from sktime.decomposition.kvadnet import gramian_gauss as gg_kvadnet
        from sktime.decomposition.kvad import gramian_gauss as gg_kvad

        Y = np.random.normal(size=(100, 5))
        Gyy_kvadnet = gg_kvadnet(torch.from_numpy(Y), sigma=sigma).cpu().numpy()
        Gyy_kvad = gg_kvad(Y, sigma=sigma)
        np.testing.assert_array_almost_equal(Gyy_kvad, Gyy_kvadnet)


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
    kvad_3d_cluster_model = kvad.kvad(ds_3d.data, ds_3d.data_lagged,
                                      Y=ds_2d.data_lagged, bandwidth=1.0)
    with torch.no_grad():
        chi_X = torch.from_numpy(ds_3d.data)
        chi_Y = torch.from_numpy(ds_3d.data_lagged)
        Y = torch.from_numpy(ds_2d.data_lagged)
        score_net = kvadnet.kvad_score(chi_X, chi_Y, Y, 1.0, mode='clamp', epsilon=1e-10).numpy()
    np.testing.assert_almost_equal(score_net, kvad_3d_cluster_model.score, decimal=1)
