import pytest
from numpy.testing import assert_array_almost_equal
from torch.utils.data import DataLoader

from deeptime.covariance import Covariance

pytest.importorskip("torch")

import torch
import torch.nn as nn
import numpy as np
import deeptime as dt

from deeptime.util.pytorch import MLP, create_timelagged_data_loader


@pytest.fixture
def two_state_hmm():
    length = 5000
    batch_size = 100
    transition_matrix = np.asarray([[0.9, 0.1], [0.1, 0.9]])
    msm = dt.markov.msm.MarkovStateModel(transition_matrix)
    dtraj = msm.simulate(length)
    traj = np.random.randn(len(dtraj))
    traj[np.where(dtraj == 1)[0]] += 20.0
    traj_stacked = np.vstack((traj, np.zeros(len(traj))))
    phi = np.random.rand() * 2.0 * np.pi
    rot = np.asarray([
        [np.cos(phi), -np.sin(phi)],
        [np.sin(phi), np.cos(phi)]])
    traj_rot = np.dot(rot, traj_stacked).T

    return traj, traj_rot, create_timelagged_data_loader(traj_rot, lagtime=1, batch_size=batch_size)


def test_tae_sanity(two_state_hmm):
    traj, traj_rot, loader = two_state_hmm
    enc = MLP([2, 2, 1], initial_batchnorm=False, nonlinearity=nn.Tanh)
    dec = MLP([1, 2, 2], initial_batchnorm=False, nonlinearity=nn.Tanh)
    tae = dt.decomposition.TAE(enc, dec, learning_rate=1e-3)
    tae.fit(loader, n_epochs=20)
    out = tae.transform(traj_rot).reshape((-1, 1))
    out = Covariance().fit(out).fetch_model().whiten(out)
    dtraj = dt.clustering.Kmeans(2).fit(out).transform(out)
    msm = dt.markov.msm.MaximumLikelihoodMSM().fit_from_discrete_timeseries(dtraj, 1).fetch_model()
    assert_array_almost_equal(msm.transition_matrix, [[.9, .1], [.1, .9]], decimal=1)
