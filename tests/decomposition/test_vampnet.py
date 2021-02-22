import pytest
from numpy.testing import assert_raises, assert_equal, assert_

pytest.importorskip("torch")

import torch
import torch.nn as nn
import numpy as np
import deeptime
from deeptime.clustering import KMeans
from deeptime.decomposition import VAMP
from deeptime.decomposition.deep import sym_inverse, covariances, vamp_score, VAMPNet, vampnet_loss
from deeptime.markov.msm import MaximumLikelihoodMSM
from deeptime.util.torch import create_timelagged_data_loader, MLP


@pytest.mark.parametrize('mode', ["trunc"])
def test_inverse_spd(fixed_seed, mode):
    X = np.random.normal(size=(15, 5))
    spd = X @ X.T  # rank at most 5
    spd_inv_qr = deeptime.numeric.spd_inv(spd)
    with torch.no_grad():
        spd_tensor = torch.from_numpy(spd)
        spd_inv = sym_inverse(spd_tensor, epsilon=1e-6, return_sqrt=False, mode=mode)
        np.testing.assert_array_almost_equal(spd_inv.numpy(), spd_inv_qr)


@pytest.mark.parametrize("remove_mean", [True, False], ids=lambda x: f"remove_mean={x}")
def test_covariances(fixed_seed, remove_mean):
    data = deeptime.data.ellipsoids().observations(1000, n_dim=5)
    tau = 10
    data_instantaneous = data[:-tau].astype(np.float64)
    data_shifted = data[tau:].astype(np.float64)
    cov_est = deeptime.covariance.Covariance(lagtime=tau, compute_c0t=True, compute_ctt=True,
                                           remove_data_mean=remove_mean)
    reference_covs = cov_est.fit(data).fetch_model()
    with torch.no_grad():
        c00, c0t, ctt = covariances(torch.from_numpy(data_instantaneous), torch.from_numpy(data_shifted),
                                    remove_mean=remove_mean)
        np.testing.assert_array_almost_equal(reference_covs.cov_00, c00.numpy())
        np.testing.assert_array_almost_equal(reference_covs.cov_0t, c0t.numpy())
        np.testing.assert_array_almost_equal(reference_covs.cov_tt, ctt.numpy())


@pytest.mark.parametrize('method', [1, 2, "E"])
@pytest.mark.parametrize('mode', ["trunc", "regularize", "clamp"])
def test_score(fixed_seed, method, mode):
    data = deeptime.data.ellipsoids(seed=13).observations(1000, n_dim=5)
    tau = 10

    vamp_model = deeptime.decomposition.VAMP(lagtime=tau, dim=None).fit(data).fetch_model()
    with torch.no_grad():
        data_instantaneous = torch.from_numpy(data[:-tau].astype(np.float64))
        data_shifted = torch.from_numpy(data[tau:].astype(np.float64))
        score_value = vamp_score(data_instantaneous, data_shifted, method=f"VAMP{method}", mode=mode)
        if method == 'E':
            # less precise due to svd implementation of torch
            np.testing.assert_array_almost_equal(score_value.numpy(), vamp_model.score(method), decimal=2)
        else:
            np.testing.assert_array_almost_equal(score_value.numpy(), vamp_model.score(method))


@pytest.mark.xfail(reason="May spuriously fail because of nondeterministic optimization of the NN", strict=False)
def test_estimator(fixed_seed):
    data = deeptime.data.ellipsoids()
    obs = data.observations(60000, n_dim=10).astype(np.float32)

    # set up the lobe
    lobe = nn.Sequential(nn.Linear(10, 1), nn.Tanh())
    # train the lobe
    opt = torch.optim.Adam(lobe.parameters(), lr=5e-4)
    for _ in range(50):
        for X, Y in deeptime.data.timeshifted_split(obs, lagtime=1, chunksize=512):
            opt.zero_grad()
            lval = vampnet_loss(lobe(torch.from_numpy(X)), lobe(torch.from_numpy(Y)))
            lval.backward()
            opt.step()

    # now let's compare
    lobe.eval()
    loader = create_timelagged_data_loader(obs, lagtime=1, batch_size=512)
    vampnet = VAMPNet(lobe=lobe)
    vampnet_model = vampnet.fit(loader).fetch_model()
    # reference model w/o learnt featurization
    projection = VAMP(lagtime=1, observable_transform=vampnet_model).fit(obs).transform(obs, propagate=True)

    dtraj = KMeans(2).fit(projection).transform(projection)
    msm_vampnet = MaximumLikelihoodMSM().fit(dtraj, lagtime=1).fetch_model()

    np.testing.assert_array_almost_equal(msm_vampnet.transition_matrix, data.msm.transition_matrix, decimal=2)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_estimator_fit(fixed_seed, dtype):
    data = deeptime.data.ellipsoids()
    obs = data.observations(60000, n_dim=2).astype(dtype)
    train, val = torch.utils.data.random_split(deeptime.data.TimeLaggedDataset.from_trajectory(1, obs), [50000, 9999])

    # set up the lobe
    linear_layer = nn.Linear(2, 1)
    lobe = nn.Sequential(linear_layer, nn.Tanh())

    with torch.no_grad():
        linear_layer.weight[0, 0] = -0.3030
        linear_layer.weight[0, 1] = 0.3060
        linear_layer.bias[0] = -0.7392

    net = VAMPNet(lobe=lobe, dtype=dtype, learning_rate=1e-8)
    train_loader = create_timelagged_data_loader(train, lagtime=1, batch_size=512)
    val_loader = create_timelagged_data_loader(val, lagtime=1, batch_size=512)
    net.fit(train_loader, n_epochs=1, validation_data=val_loader, validation_score_callback=lambda *x: x)

    # reference model w/o learnt featurization
    projection = VAMP(lagtime=1, observable_transform=net).fit(obs).fetch_model().transform(obs)

    dtraj = KMeans(2).fit(projection).transform(projection)
    msm_vampnet = MaximumLikelihoodMSM().fit(dtraj, lagtime=1).fetch_model()

    np.testing.assert_array_almost_equal(msm_vampnet.transition_matrix, data.msm.transition_matrix, decimal=2)


def test_mlp_sanity():
    mlp = MLP([100, 10, 2])
    with torch.no_grad():
        x = torch.empty((5, 100)).normal_()
        mlp(x)


def test_no_side_effects():
    mlp = nn.Linear(10, 2)
    data = deeptime.data.ellipsoids()
    obs = data.observations(100, n_dim=10).astype(np.float32)
    net = VAMPNet(lobe=mlp, dtype=np.float32, learning_rate=1e-8)
    train_loader = create_timelagged_data_loader(obs, lagtime=1, batch_size=512)
    model1 = net.fit(train_loader, n_epochs=1).fetch_model()
    model2 = net.fit(train_loader, n_epochs=1).fetch_model()
    with torch.no_grad():
        assert_(model1.lobe is not model2.lobe)  # check it is not the same instance
