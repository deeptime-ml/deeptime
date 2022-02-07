"""
Chapman-Kolmogorov test
=======================

Tests the Chapman-Kolmogorov equality, see :meth:`ck_test <deeptime.util.validation.ck_test>`.

We demonstrate how to re-use the view grid to overlay two CK tests.
"""
from deeptime.clustering import KMeans
from deeptime.data import ellipsoids
from deeptime.markov import TransitionCountEstimator
from deeptime.markov.msm import BayesianMSM
from deeptime.plots.chapman_kolmogorov import plot_ck_test

traj = ellipsoids().observations(500)
traj2 = ellipsoids().observations(500)
dtraj = KMeans(n_clusters=15).fit_transform(traj)
dtraj2 = KMeans(n_clusters=15).fit_transform(traj2)

models = []
models2 = []
for lag in [2, 3, 5, 13]:
    counts_estimator = TransitionCountEstimator(lagtime=lag, count_mode='effective')
    models.append(BayesianMSM().fit_fetch(counts_estimator.fit_fetch(dtraj).submodel_largest()))
    models2.append(BayesianMSM().fit_fetch(counts_estimator.fit_fetch(dtraj2).submodel_largest()))

test_model = models[0]
ck_test = test_model.ck_test(models, n_metastable_sets=2)
grid = plot_ck_test(ck_test, legend=False)

test_model2 = models2[0]
ck_test = test_model2.ck_test(models2, n_metastable_sets=2)
plot_ck_test(ck_test, legend=True, grid=grid)
