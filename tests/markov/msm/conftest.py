import collections

import numpy as np
import pytest

import deeptime
from deeptime.markov import TransitionCountEstimator
from deeptime.markov.msm import MaximumLikelihoodMSM, AugmentedMSMEstimator
from tests.markov.msm.util import _make_reference_data, MLMSM_PARAMS, MLMSM_IDS, AMM_IDS, AMM_PARAMS


@pytest.fixture(scope="module")
def reference_data():
    return _make_reference_data


@pytest.fixture(scope="module")
def make_double_well_data():
    def _make_double_well_data(msm_type, reversible, statdist_constraint, sparse, count_mode):
        return _make_reference_data("doublewell", msm_type, reversible, statdist_constraint, sparse, count_mode)

    return _make_double_well_data


@pytest.fixture(scope="module", params=MLMSM_PARAMS, ids=MLMSM_IDS)
def double_well_mlmsm(request):
    msm_type, reversible, statdist_constraint, sparse = request.param
    scenario = _make_reference_data(
        dataset="doublewell", msm_type=msm_type, reversible=reversible,
        statdist_constraint=statdist_constraint, sparse=sparse, count_mode="sliding")
    return scenario


@pytest.fixture(scope="module", params=AMM_PARAMS, ids=AMM_IDS)
def double_well_amm(request):
    msm_type, count_mode = request.param
    return _make_reference_data(dataset="doublewell", msm_type=msm_type, reversible=True, statdist_constraint=False,
                                sparse=False, count_mode=count_mode)


@pytest.fixture(scope="module", params=MLMSM_PARAMS+AMM_PARAMS, ids=MLMSM_IDS+AMM_IDS)
def double_well_all(request):
    if request.param[0] == "AMM":
        msm_type, count_mode = request.param
        return _make_reference_data(dataset="doublewell", msm_type=msm_type, reversible=True, statdist_constraint=False,
                                    sparse=False, count_mode=count_mode)
    if request.param[0] == "MLMSM":
        msm_type, reversible, statdist_constraint, sparse = request.param
        return _make_reference_data(
            dataset="doublewell", msm_type=msm_type, reversible=reversible,
            statdist_constraint=statdist_constraint, sparse=sparse, count_mode="sliding")
    raise ValueError("unknown request param {}".format(request.param[0]))

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
    dtraj = deeptime.data.double_well_discrete().dtraj
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

    amm_expectations = np.linspace(0.01, 2. * np.pi, 66).reshape(-1, 1) ** 0.5
    amm_m = np.array([1.9])
    amm_w = np.array([2.0])
    amm_sigmas = 1. / np.sqrt(2) / np.sqrt(amm_w)
    amm_sd = list(set(double_well.dtraj))

    amm_ftraj = amm_expectations[[amm_sd.index(d) for d in double_well.dtraj], :]
    est_amm = AugmentedMSMEstimator.estimator_from_feature_trajectories(double_well.dtraj, amm_ftraj,
                                                                        n_states=np.max(double_well.dtraj)+1,
                                                                        experimental_measurements=amm_m,
                                                                        sigmas=amm_sigmas)
    count_model = TransitionCountEstimator(lagtime=double_well.lagtime, count_mode="sliding", sparse=False) \
        .fit(double_well.dtraj).fetch_model()
    count_model = count_model.submodel_largest()
    amm = est_amm.fit(count_model)

    def get(mode, **kw):
        r""" Yields the scenario as well as estimator and msm
        estimated with flags reversible / fixed statdist / sparse. """
        if mode == "MLMSM":
            reversible, statdist_constraint, sparse = kw['reversible'], kw['statdist_constraint'], kw['sparse']
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
        elif mode == "amm":
            return double_well, est_amm, amm

    return get


@pytest.fixture(scope="module", params=MLMSM_PARAMS, ids=MLMSM_IDS)
def double_well_msm_all(request, double_well_msm):
    mode, reversible, statdist_constraint, sparse = request.param
    return double_well_msm(mode, reversible=reversible, statdist_constraint=statdist_constraint, sparse=sparse)


@pytest.fixture(scope="module", params=MLMSM_PARAMS + AMM_PARAMS, ids=MLMSM_IDS + AMM_IDS)
def double_well_msm_amm_all(request, double_well_msm):
    mode, reversible, statdist_constraint, sparse = request.param
    return double_well_msm(mode, reversible=reversible, statdist_constraint=statdist_constraint, sparse=sparse)


@pytest.fixture(scope="module", params=[
    (True, False, False), (False, False, False),
    (True, False, True), (False, False, True)
], ids=["reversible", "nonreversible",
        "reversible_sparse", "nonreversible_sparse"])
def double_well_msm_nostatdist_constraint(request, double_well_msm):
    reversible, statdist_constraint, sparse = request.param
    return double_well_msm(reversible=reversible, statdist_constraint=statdist_constraint, sparse=sparse)
