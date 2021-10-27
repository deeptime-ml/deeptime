import numpy as np
import pytest
import deeptime as dt
from numpy.testing import assert_allclose, assert_equal, assert_raises


def test_invalid_mlags():
    data = dt.data.double_well_discrete().dtraj
    est = dt.markov.msm.MaximumLikelihoodMSM()
    est.fit(data, lagtime=1)
    with assert_raises(ValueError):
        est.chapman_kolmogorov_validator(2, mlags=[0, 1, -10])


@pytest.mark.parametrize("n_jobs", [1, 2], ids=lambda x: f"n_jobs={x}")
@pytest.mark.parametrize("mlags", [2, [0, 1, 10], [1, 10]], ids=lambda x: f"mlags={x}")
@pytest.mark.parametrize("estimator_type", ["MLMSM", "BMSM", "HMM", "BHMM"])
def test_cktest_double_well(estimator_type, n_jobs, mlags):
    # maximum-likelihood estimates
    estref = np.array([[[1., 0.],
                        [0., 1.]],
                       [[0.89806859, 0.10193141],
                        [0.10003466, 0.89996534]],
                       [[0.64851782, 0.35148218],
                        [0.34411751, 0.65588249]]])
    predref = np.array([[[1., 0.],
                         [0., 1.]],
                        [[0.89806859, 0.10193141],
                         [0.10003466, 0.89996534]],
                        [[0.62613723, 0.37386277],
                         [0.3669059, 0.6330941]]])
    dtraj = dt.data.double_well_discrete().dtraj_n6good
    if estimator_type == "MLMSM":
        est = dt.markov.msm.MaximumLikelihoodMSM()
        est.fit(dtraj, lagtime=1)
        validator = est.chapman_kolmogorov_validator(2, mlags=mlags)
    elif estimator_type == "BMSM":
        bmsm_est = dt.markov.msm.BayesianMSM()
        counts = dt.markov.TransitionCountEstimator(1, "effective").fit(dtraj).fetch_model().submodel_largest()
        bmsm_est.fit(counts)
        validator = bmsm_est.chapman_kolmogorov_validator(2, mlags=mlags)
    elif estimator_type == "HMM":
        hmm_init = dt.markov.hmm.init.discrete.metastable_from_data(dtraj, 2, lagtime=1)
        hmm_est = dt.markov.hmm.MaximumLikelihoodHMM(hmm_init, lagtime=1)
        hmm_test = hmm_est.fit(dtraj).fetch_model()
        validator = hmm_est.chapman_kolmogorov_validator(mlags, hmm_test.submodel_largest(dtrajs=dtraj))
    elif estimator_type == "BHMM":
        bhmm_est = dt.markov.hmm.BayesianHMM.default(dtraj, 2, lagtime=1)
        bhmm = bhmm_est.fit(dtraj).fetch_model().submodel_largest(dtrajs=dtraj)
        validator = bhmm_est.chapman_kolmogorov_validator(mlags, bhmm)
    else:
        pytest.fail()

    validator.err_est = True
    cktest = validator.fit(dtraj).fetch_model()
    if not isinstance(mlags, list):
        assert_equal(cktest.lagtimes, [0, 1])
    else:
        assert_equal(cktest.lagtimes, mlags)

    ix = []
    if 0 in cktest.lagtimes:
        ix.append(0)
    if 1 in cktest.lagtimes:
        ix.append(1)
    if 10 in cktest.lagtimes:
        ix.append(2)
    ix = np.array(ix, dtype=int)

    assert_allclose(cktest.estimates, estref[ix], rtol=.1, atol=10.)
    assert_allclose(cktest.predictions, predref[ix], rtol=.1, atol=10.)
