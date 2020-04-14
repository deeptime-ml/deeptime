import collections

import numpy as np
import pytest

import sktime
from sktime.markov import TransitionCountEstimator
from sktime.markov.msm import MaximumLikelihoodMSM

DoubleWellScenario = collections.namedtuple("DoubleWellScenario", [
    "dtraj", "lagtime", "stationary_distribution", "n_states", "selected_count_fraction"
])


def estimate_markov_model(dtrajs, lag, **kw):
    statdist_constraint = kw.pop('statdist', None)
    connectivity = kw.pop('connectivity_threshold', 0.)
    sparse = kw.pop('sparse', False)
    count_model = TransitionCountEstimator(lagtime=lag, count_mode="sliding", sparse=sparse).fit(dtrajs).fetch_model()
    count_model = count_model.submodel_largest(probability_constraint=statdist_constraint,
                                               connectivity_threshold=connectivity)
    est = MaximumLikelihoodMSM(stationary_distribution_constraint=statdist_constraint, sparse=sparse, **kw)
    est.fit(count_model)
    return est, est.fetch_model()


@pytest.fixture(scope="module")
def double_well():
    dtraj = sktime.data.datasets.double_well_discrete().dtraj
    nu = 1. * np.bincount(dtraj)
    return DoubleWellScenario(dtraj=dtraj, lagtime=10, stationary_distribution=nu / nu.sum(), n_states=66,
                              selected_count_fraction=1.)


@pytest.fixture(scope="module")
def double_well_msm(double_well):
    maxerr = 1e-12
    estrev, msmrev = estimate_markov_model(double_well.dtraj, double_well.lagtime, maxerr=maxerr)
    estrevpi, msmrevpi = estimate_markov_model(double_well.dtraj, double_well.lagtime, maxerr=maxerr,
                                               statdist=double_well.stationary_distribution)
    est, msm = estimate_markov_model(double_well.dtraj, double_well.lagtime, reversible=False, maxerr=maxerr)

    estrev_sparse, msmrev_sparse = estimate_markov_model(double_well.dtraj, double_well.lagtime, sparse=True,
                                                         maxerr=maxerr)
    estrevpi_sparse, msmrevpi_sparse = estimate_markov_model(double_well.dtraj, double_well.lagtime, maxerr=maxerr,
                                                             statdist=double_well.stationary_distribution,
                                                             sparse=True)
    est_sparse, msm_sparse = estimate_markov_model(double_well.dtraj, double_well.lagtime, reversible=False,
                                                   sparse=True,
                                                   maxerr=maxerr)

    def get(reversible, statdist_constraint, sparse):
        r""" Yields the scenario as well as estimator and msm
        estimated with flags reversible / fixed statdist / sparse. """
        if reversible:
            if statdist_constraint:
                if sparse:
                    return double_well, estrevpi_sparse, msmrevpi_sparse
                else:
                    return double_well, estrevpi, msmrevpi
            else:
                if sparse:
                    return double_well, estrev_sparse, msmrev_sparse
                else:
                    return double_well, estrev, msmrev
        else:
            if statdist_constraint:
                raise ValueError("nonreversible mle with fixed stationary distribution not implemented")
            else:
                if sparse:
                    return double_well, est_sparse, msm_sparse
                else:
                    return double_well, est, msm

    return get


@pytest.fixture(scope="module", params=[
    (True, False, False), (True, True, False), (False, False, False),
    (True, False, True), (True, True, True), (False, False, True)
], ids=["reversible", "reversible_pi", "nonreversible",
        "reversible_sparse", "reversible_pi_sparse", "nonreversible_sparse"])
def double_well_msm_all(request, double_well_msm):
    reversible, statdist_constraint, sparse = request.param
    return double_well_msm(reversible=reversible, statdist_constraint=statdist_constraint, sparse=sparse)


@pytest.fixture(scope="module", params=[
    (True, False, False), (False, False, False),
    (True, False, True), (False, False, True)
], ids=["reversible", "nonreversible",
        "reversible_sparse", "nonreversible_sparse"])
def double_well_msm_nostatdist_constraint(request, double_well_msm):
    reversible, statdist_constraint, sparse = request.param
    return double_well_msm(reversible=reversible, statdist_constraint=statdist_constraint, sparse=sparse)
