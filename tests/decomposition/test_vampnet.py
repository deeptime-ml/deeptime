import pytest
pytest.importorskip("torch")

import torch
import torch.nn as nn
import numpy as np
import sktime
from sktime.clustering import KmeansClustering
from sktime.decomposition import VAMP
from sktime.decomposition.vampnet import sym_inverse, covariances, score, VAMPNet, loss
from sktime.markov.msm import MaximumLikelihoodMSM


def test_inverse_spd():
    X = np.random.normal(size=(15, 5))
    spd = X @ X.T  # rank at most 5
    spd_inv_qr = sktime.numeric.spd_inv(spd)
    with torch.no_grad():
        spd_tensor = torch.from_numpy(spd)
        spd_inv = sym_inverse(spd_tensor, epsilon=1e-6, ret_sqrt=False)
        np.testing.assert_array_almost_equal(spd_inv.numpy(), spd_inv_qr)


@pytest.mark.parametrize("remove_mean", [True, False], ids=lambda x: f"remove_mean={x}")
def test_covariances(remove_mean):
    data = sktime.data.ellipsoids().observations(1000, n_dim=5)
    tau = 10
    data_instantaneous = data[:-tau].astype(np.float64)
    data_shifted = data[tau:].astype(np.float64)
    cov_est = sktime.covariance.Covariance(lagtime=tau, compute_c0t=True, compute_ctt=True,
                                           remove_data_mean=remove_mean)
    reference_covs = cov_est.fit(data).fetch_model()
    with torch.no_grad():
        c00, c0t, ctt = covariances(torch.from_numpy(data_instantaneous), torch.from_numpy(data_shifted),
                                    remove_mean=remove_mean)
        np.testing.assert_array_almost_equal(reference_covs.cov_00, c00.numpy())
        np.testing.assert_array_almost_equal(reference_covs.cov_0t, c0t.numpy())
        np.testing.assert_array_almost_equal(reference_covs.cov_tt, ctt.numpy())


@pytest.mark.parametrize('method', ["VAMP1", "VAMP2"])
def test_score(method):
    data = sktime.data.ellipsoids().observations(1000, n_dim=5)
    tau = 10

    vamp_model = sktime.decomposition.VAMP(lagtime=tau, dim=None).fit(data).fetch_model()
    with torch.no_grad():
        data_instantaneous = torch.from_numpy(data[:-tau].astype(np.float64))
        data_shifted = torch.from_numpy(data[tau:].astype(np.float64))
        score_value = score(data_instantaneous, data_shifted, method=method)
        np.testing.assert_array_almost_equal(score_value.numpy(), vamp_model.score(score_method=method))


def test_estimator():
    data = sktime.data.ellipsoids()
    obs = data.observations(60000, n_dim=10).astype(np.float32)

    # set up the lobe
    lobe = nn.Sequential(nn.Linear(10, 1), nn.Tanh())
    # train the lobe
    opt = torch.optim.Adam(lobe.parameters(), lr=5e-4)
    for _ in range(50):
        for X, Y in sktime.data.timeshifted_split(obs, lagtime=1, chunksize=512):
            opt.zero_grad()
            lval = loss(lobe(torch.from_numpy(X)), lobe(torch.from_numpy(Y)))
            lval.backward()
            opt.step()

    # now let's compare
    lobe.eval()
    vampnet = VAMPNet(1, lobe=lobe)
    vampnet_model = vampnet.fit(obs).fetch_model()
    # np.testing.assert_array_less(vamp_model.timescales()[0], vampnet_model.timescales()[0])

    projection = vampnet_model.transform(obs)
    # reference model w/o learnt featurization
    projection = VAMP(lagtime=1).fit(projection).fetch_model().transform(projection)

    dtraj = KmeansClustering(2).fit(projection).transform(projection)
    msm_vampnet = MaximumLikelihoodMSM().fit(dtraj, lagtime=1).fetch_model()

    np.testing.assert_array_almost_equal(msm_vampnet.transition_matrix, data.msm.transition_matrix, decimal=2)

def test_estimator_fit():
    data = sktime.data.ellipsoids()
    obs = data.observations(60000, n_dim=10).astype(np.float32)
    train, val = torch.utils.data.random_split(sktime.data.TimeSeriesDataset(obs, lagtime=1), [50000, 9999])

    # set up the lobe
    lobe = nn.Sequential(nn.Linear(10, 1), nn.Tanh())

    net = VAMPNet(lagtime=1, lobe=lobe)
    net.fit(train, n_epochs=2, batch_size=128, validation_data=val)
    net_model = net.fetch_model()

    projection = net_model.transform(obs)
    # reference model w/o learnt featurization
    projection = VAMP(lagtime=1).fit(projection).fetch_model().transform(projection)

    dtraj = KmeansClustering(2).fit(projection).transform(projection)
    msm_vampnet = MaximumLikelihoodMSM().fit(dtraj, lagtime=1).fetch_model()

    np.testing.assert_array_almost_equal(msm_vampnet.transition_matrix, data.msm.transition_matrix, decimal=2)
