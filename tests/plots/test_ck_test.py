import warnings

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest

from deeptime.clustering import KMeans
from deeptime.data import ellipsoids
from deeptime.decomposition import VAMP
from deeptime.plots.chapman_kolmogorov import plot_ck_test
from tests.testing_utilities import estimate_markov_model


@pytest.mark.parametrize("hidden", [False, True], ids=lambda x: f"hidden={x}")
@pytest.mark.parametrize("bayesian", [False, True], ids=lambda x: f"bayesian={x}")
def test_sanity_msm(hidden, bayesian):
    mlags = np.arange(1, 5)
    traj = ellipsoids().observations(20000)
    dtraj = KMeans(n_clusters=15).fit_transform(traj)
    models = []
    for lag in mlags:
        models.append(estimate_markov_model(lag, dtraj, hidden, bayesian))
    test_model = models[0]
    if not hidden:
        cktest = test_model.ck_test(models, n_metastable_sets=2)
    else:
        cktest = test_model.ck_test(models)
    cktest.plot(conf=1)


@pytest.mark.parametrize("fractional", [False, True])
def test_sanity_vamp(fractional):
    traj = ellipsoids().observations(20000)
    models = []
    if fractional:
        lags = [2, 3, 4, 5]
    else:
        lags = [2, 4, 6, 8]
    for lag in lags:
        models.append(VAMP(lag, dim=2).fit_fetch(traj))

    if fractional:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.ComplexWarning)
            plot_ck_test(models[0].ck_test(models))
    else:
        plot_ck_test(models[0].ck_test(models))
