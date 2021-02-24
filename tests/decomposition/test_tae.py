import pytest
from numpy.testing import assert_array_almost_equal

pytest.importorskip("torch")

from deeptime.covariance import Covariance
from deeptime.decomposition.deep import TVAEEncoder

import torch.nn as nn
import numpy as np
import deeptime as dt

from deeptime.util.torch import MLP, create_timelagged_data_loader


@pytest.fixture
def two_state_hmm():
    length = 1000
    batch_size = 64
    transition_matrix = np.asarray([[0.9, 0.1], [0.1, 0.9]])
    msm = dt.markov.msm.MarkovStateModel(transition_matrix)
    dtraj = msm.simulate(length, seed=42)
    traj = np.random.randn(len(dtraj))
    traj[np.where(dtraj == 1)[0]] += 20.0
    traj_stacked = np.vstack((traj, np.zeros(len(traj))))
    phi = np.random.rand() * 2.0 * np.pi
    rot = np.asarray([
        [np.cos(phi), -np.sin(phi)],
        [np.sin(phi), np.cos(phi)]])
    traj_rot = np.dot(rot, traj_stacked).T

    return traj, traj_rot, create_timelagged_data_loader(traj_rot, lagtime=1, batch_size=batch_size)


def setup_tae():
    enc = MLP([2, 1], initial_batchnorm=False, nonlinearity=nn.Tanh)
    dec = MLP([1, 2], initial_batchnorm=False, nonlinearity=nn.Tanh)
    return dt.decomposition.deep.TAE(enc, dec, learning_rate=1e-3)


def setup_tvae():
    enc = TVAEEncoder([2, 1], nonlinearity=nn.ReLU)
    dec = MLP([1, 2], initial_batchnorm=False, nonlinearity=nn.ReLU)
    return dt.decomposition.deep.TVAE(enc, dec, learning_rate=1e-3)


@pytest.mark.parametrize('model', ['tae', 'tvae'])
def test_sanity(fixed_seed, two_state_hmm, model):
    traj, traj_rot, loader = two_state_hmm
    tae = setup_tae() if model == 'tae' else setup_tvae()
    tae.fit(loader, n_epochs=40)
    out = tae.transform(traj_rot).reshape((-1, 1))
    out = Covariance().fit(out).fetch_model().whiten(out)
    dtraj = dt.clustering.KMeans(2).fit(out).transform(out)
    msm = dt.markov.msm.MaximumLikelihoodMSM().fit_from_discrete_timeseries(dtraj, 1).fetch_model()
    assert_array_almost_equal(msm.transition_matrix, [[.9, .1], [.1, .9]], decimal=1)
