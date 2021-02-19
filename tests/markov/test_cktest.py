import numpy as np
import pytest
import deeptime as dt
from numpy.testing import assert_allclose


@pytest.mark.parametrize("n_jobs", [1, 2], ids=lambda x: f"n_jobs={x}")
@pytest.mark.parametrize("estimator_type", ["MLMSM", "BMSM", "HMM", "BHMM"])
@pytest.mark.parametrize("mlags", [[1, 10]])
def test_cktest_double_well(estimator_type, n_jobs, mlags):
    # maximum-likelihood estimates
    estref = np.array([[[0.89806859, 0.10193141],
                        [0.10003466, 0.89996534]],
                       [[0.64851782, 0.35148218],
                        [0.34411751, 0.65588249]]])
    predref = np.array([[[0.89806859, 0.10193141],
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
        hmm_est = dt.markov.hmm.MaximumLikelihoodHMM(hmm_init)
        hmm_test = hmm_est.fit(dtraj).fetch_model()
        validator = hmm_est.chapman_kolmogorov_validator(mlags, hmm_test.submodel_largest(dtrajs=dtraj))
    elif estimator_type == "BHMM":
        bhmm_est = dt.markov.hmm.BayesianHMM.default(dtraj, 2, lagtime=1)
        bhmm = bhmm_est.fit(dtraj).fetch_model().submodel_largest(dtrajs=dtraj)
        validator = bhmm_est.chapman_kolmogorov_validator(mlags, bhmm)
    else:
        pytest.fail()

    cktest = validator.fit(dtraj).fetch_model()
    # rough agreement with MLE
    assert_allclose(cktest.estimates, estref, rtol=.1, atol=10.)
    assert_allclose(cktest.predictions, predref, rtol=.1, atol=10.)
