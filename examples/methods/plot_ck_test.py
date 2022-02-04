"""
Chapman-Kolmogorov test
=======================

Tests the Chapman-Kolmogorov equality, see :meth:`ck_test <deeptime.validation.ck_test>`.
"""
from deeptime.clustering import KMeans
from deeptime.data import ellipsoids
from deeptime.markov import TransitionCountEstimator
from deeptime.markov.msm import BayesianMSM
from deeptime.plots.chapman_kolmogorov import plot_ck_test

traj = ellipsoids().observations(2000)
dtraj = KMeans(n_clusters=15).fit_transform(traj)

models = []
for lag in [2, 3, 5, 13]:
    counts = TransitionCountEstimator(lagtime=lag, count_mode='effective').fit_fetch(dtraj).submodel_largest()
    models.append(BayesianMSM().fit_fetch(counts))

test_model = models[0]
ck_test = test_model.ck_test(models, n_metastable_sets=2)
plot_ck_test(ck_test)
