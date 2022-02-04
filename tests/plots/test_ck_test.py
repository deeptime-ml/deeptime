import numpy as np
import pytest

from deeptime.clustering import KMeans
from deeptime.data import ellipsoids
from deeptime.markov.hmm import MaximumLikelihoodHMM, BayesianHMM
from deeptime.markov.hmm.init import discrete
from deeptime.markov.msm import MaximumLikelihoodMSM, BayesianMSM
from deeptime.plots.chapman_kolmogorov import plot_ck_test


@pytest.mark.parametrize("hidden", [False, True], ids=lambda x: f"hidden={x}")
@pytest.mark.parametrize("bayesian", [False, True], ids=lambda x: f"bayesian={x}")
def test_sanity_msm(hidden, bayesian):
    mlags = np.arange(1, 5)
    traj = ellipsoids().observations(20000)
    dtraj = KMeans(n_clusters=15).fit_transform(traj)
    models = []
    for lag in mlags:
        if not hidden:
            msm = MaximumLikelihoodMSM(lagtime=lag).fit_fetch(dtraj, count_mode='effective')
            if bayesian:
                msm = BayesianMSM().fit_fetch(msm)
            models.append(msm)
        else:
            if not bayesian:
                hmm_init = discrete.metastable_from_data(dtraj, n_hidden_states=2, lagtime=lag)
                hmm = MaximumLikelihoodHMM(hmm_init, lagtime=lag).fit_fetch(dtraj)
                models.append(hmm)
            else:
                bhmm = BayesianHMM.default(n_hidden_states=2, lagtime=lag, dtrajs=dtraj).fit_fetch(dtraj)
                models.append(bhmm)
    test_model = models[0]
    if not hidden:
        cktest = test_model.cktest(models, n_metastable_sets=2)
    else:
        cktest = test_model.cktest(models)
    plot_ck_test(cktest, conf=1)
