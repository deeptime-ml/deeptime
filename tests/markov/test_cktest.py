import numpy as np
import pytest
import deeptime as dt
from flaky import flaky
from numpy.testing import assert_allclose, assert_equal, assert_raises

from tests.testing_utilities import estimate_markov_model


def test_invalid_mlags():
    data = dt.data.double_well_discrete().dtraj
    est = dt.markov.msm.MaximumLikelihoodMSM()
    est.fit(data, lagtime=1)
    with assert_raises(ValueError):
        est.chapman_kolmogorov_validator(2, mlags=[0, 1, -10])


@flaky(max_runs=3, min_passes=1)
@pytest.mark.parametrize("lagtimes", [[1], [1, 10]], ids=lambda x: f"lagtimes={x}")
@pytest.mark.parametrize("hidden", [False, True])
@pytest.mark.parametrize("bayesian", [False, True])
def test_cktest_double_well(hidden, bayesian, lagtimes):
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

    models = []
    for lag in lagtimes:
        models.append(estimate_markov_model(lag, dtraj, hidden=hidden, bayesian=bayesian, n_hidden=2))

    if hidden:
        cktest = models[0].ck_test(models=models, err_est=True)
    else:
        cktest = models[0].ck_test(models=models, n_metastable_sets=2)
    assert_equal(cktest.lagtimes, [0] + lagtimes)

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
