from copy import deepcopy

import pytest
from flaky import flaky

pytest.importorskip("torch")

from numpy.testing import assert_, assert_almost_equal
from torch.utils.data import DataLoader

from deeptime.util.data import TrajectoryDataset

import torch
import torch.nn as nn
import numpy as np
import deeptime
from deeptime.clustering import KMeans
from deeptime.decomposition import VAMP, vamp_score_data
from deeptime.decomposition.deep import sym_inverse, covariances, vamp_score, VAMPNet, vampnet_loss
from deeptime.markov.msm import MaximumLikelihoodMSM
from deeptime.util.torch import MLP


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
        score_value_ref = vamp_score_data(data_instantaneous.numpy(), data_shifted.numpy(), r=method)
        if method == 'E':
            # less precise due to svd implementation of torch
            assert_almost_equal(score_value.numpy(), vamp_model.score(method), decimal=2)
            assert_almost_equal(score_value.numpy(), score_value_ref, decimal=2)
        else:
            np.testing.assert_almost_equal(score_value.numpy(), vamp_model.score(method))
            assert_almost_equal(score_value.numpy(), score_value_ref)


@flaky(max_runs=3, min_passes=1)
def test_estimator(fixed_seed):
    data = deeptime.data.ellipsoids()
    obs = data.observations(6000, n_dim=10).astype(np.float32)

    # set up the lobe
    lobe = nn.Sequential(nn.Linear(10, 1), nn.Tanh())
    # train the lobe
    opt = torch.optim.Adam(lobe.parameters(), lr=1e-2)
    for _ in range(50):
        for X, Y in deeptime.util.data.timeshifted_split(obs, lagtime=1, chunksize=512):
            opt.zero_grad()
            lval = vampnet_loss(lobe(torch.from_numpy(X)), lobe(torch.from_numpy(Y)))
            lval.backward()
            opt.step()

    # now let's compare
    lobe.eval()
    ds = TrajectoryDataset(1, obs)
    loader = DataLoader(ds, batch_size=512)
    loader_val = DataLoader(ds, batch_size=512)
    vampnet = VAMPNet(lobe=lobe)
    vampnet_model = vampnet.fit(loader, validation_loader=loader_val).fetch_model()
    assert_(len(vampnet.train_scores) > 0)
    assert_(len(vampnet.validation_scores) > 0)
    # reference model w/o learnt featurization
    projection = VAMP(lagtime=1, observable_transform=vampnet_model).fit(obs).transform(obs, propagate=True)

    dtraj = KMeans(2).fit(projection).transform(projection)
    msm_vampnet = MaximumLikelihoodMSM().fit(dtraj, lagtime=1).fetch_model()

    np.testing.assert_array_almost_equal(msm_vampnet.transition_matrix, data.msm.transition_matrix, decimal=2)


@pytest.mark.parametrize("shared_lobe", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_estimator_fit(fixed_seed, dtype, shared_lobe):
    data = deeptime.data.ellipsoids()
    obs = data.observations(60000, n_dim=2).astype(dtype)
    train, val = torch.utils.data.random_split(TrajectoryDataset(1, obs), [50000, 9999])

    # set up the lobe
    linear_layer = nn.Linear(2, 1)
    lobe = nn.Sequential(linear_layer, nn.Tanh())

    with torch.no_grad():
        linear_layer.weight[0, 0] = -0.3030
        linear_layer.weight[0, 1] = 0.3060
        linear_layer.bias[0] = -0.7392

    if shared_lobe:
        lobe_t = None
    else:
        lobe_t = deepcopy(lobe)

    net = VAMPNet(lobe=lobe, dtype=dtype, learning_rate=1e-8, lobe_timelagged=lobe_t)
    train_loader = DataLoader(train, batch_size=512, shuffle=True)
    val_loader = DataLoader(val, batch_size=512)
    net.fit(train_loader, n_epochs=1, validation_loader=val_loader, validation_score_callback=lambda *x: x)

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
    ds = TrajectoryDataset(1, obs)
    train_loader = DataLoader(ds, batch_size=512, shuffle=True)
    model1 = net.fit(train_loader, n_epochs=1).fetch_model()
    model2 = net.fit(train_loader, n_epochs=1).fetch_model()
    with torch.no_grad():
        assert_(model1.lobe is not model2.lobe)  # check it is not the same instance


@pytest.mark.parametrize("dtype,torch_dtype", [(np.float32, torch.float32), (np.float64, torch.float64)])
def test_constructor_applies_dtype_to_lobes(dtype, torch_dtype):
    # regression: the lobes were assigned before self.device/self.dtype, so the lobe setters
    # cast/moved them using the *default* dtype/device instead of the requested ones. With
    # dtype=float64 requested, the lobe parameters must already be float64 after construction.
    lobe = nn.Linear(5, 2)
    lobe_tl = nn.Linear(5, 2)
    net = VAMPNet(lobe=lobe, lobe_timelagged=lobe_tl, dtype=dtype)
    assert_(next(net.lobe.parameters()).dtype == torch_dtype)
    assert_(next(net.lobe_timelagged.parameters()).dtype == torch_dtype)


def test_symeig_reg_regularize_is_bounded_below():
    # A round-off error of order -epsilon in a rank-deficient direction must not cancel the
    # regularization: the regularized eigenvalues are bounded from below by epsilon.
    from deeptime.decomposition.deep._vampnet import symeig_reg, sym_inverse
    eps = 1e-6
    mat = torch.diag(torch.tensor([1., -eps], dtype=torch.float64))
    eigval, _ = symeig_reg(mat, epsilon=eps, mode='regularize')
    assert torch.all(eigval >= eps)
    assert torch.isfinite(sym_inverse(mat, epsilon=eps, mode='regularize')).all()


def test_vamp_score_float32_matches_float64_for_rank_deficient_features():
    # Softmax features sum to one, so their covariance has an exact null direction. With float32
    # covariance accumulation over a large batch, the round-off in that direction is of the order of
    # epsilon and inflated the score far beyond its bound of n_features in the past.
    from deeptime.decomposition.deep import vamp_score
    torch.manual_seed(0)
    n, d = 75000, 6
    x = torch.randn(n, 3, dtype=torch.float64)
    w = torch.randn(3, d, dtype=torch.float64)
    chi = torch.softmax(x @ w, dim=1)
    chi_lagged = torch.softmax((x + 0.3 * torch.randn_like(x)) @ w, dim=1)
    ref = vamp_score(chi, chi_lagged, method='VAMP2', mode='regularize')
    out = vamp_score(chi.float(), chi_lagged.float(), method='VAMP2', mode='regularize')
    assert out.dtype == torch.float32
    assert float(out) <= d + 1e-3
    assert abs(float(out) - float(ref)) < 1e-3
