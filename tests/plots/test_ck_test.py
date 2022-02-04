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
    plot_ck_test(cktest, conf=1)


def test_sanity_vamp():
    traj = ellipsoids().observations(20000)
    models = []
    lags = [2, 3, 4, 5]
    for lag in lags:
        models.append(VAMP(lag, dim=2).fit_fetch(traj))

    ck_test = models[0].ck_test(models)
    plot_ck_test(ck_test)
