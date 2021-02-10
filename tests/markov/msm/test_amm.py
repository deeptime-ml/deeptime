r"""Unit test for the AMM module

.. moduleauthor:: S. Olsson <solsson AT zedat DOT fu DASH berlin DOT de>

"""

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_equal

from deeptime.data import birth_death_chain
from deeptime.markov import TransitionCountEstimator
from deeptime.markov.msm import MarkovStateModel, AugmentedMSMEstimator


def test_amm_sanity(fixed_seed):
    # Meta-stable birth-death chain
    b = 2
    q = np.zeros(7)
    p = np.zeros(7)
    q[1:] = 0.5
    p[0:-1] = 0.5
    q[2] = 1.0 - 10 ** (-b)
    q[4] = 10 ** (-b)
    p[2] = 10 ** (-b)
    p[4] = 1.0 - 10 ** (-b)

    bdc = birth_death_chain(q, p)
    P = bdc.transition_matrix
    dtraj = MarkovStateModel(P).simulate(n_steps=10000, start=0, seed=42)
    tau = 1

    k = 3
    # Predictions and experimental data
    E = np.vstack((np.linspace(-0.1, 1., 7), np.linspace(1.5, -0.1, 7))).T
    m = np.array([0.0, 0.0])
    w = np.array([2.0, 2.5])
    sigmas = 1. / np.sqrt(2) / np.sqrt(w)

    """ Feature trajectory """
    ftraj = E[dtraj, :]

    amm_estimator = AugmentedMSMEstimator(expectations_by_state=E, experimental_measurements=m,
                                          experimental_measurement_weights=w)
    counts = TransitionCountEstimator(lagtime=tau, count_mode="sliding").fit(dtraj).fetch_model()

    amm = amm_estimator.fit(counts).fetch_model()
    amm_convenience_estimator = AugmentedMSMEstimator.estimator_from_feature_trajectories(
        dtraj, ftraj, n_states=counts.n_states_full, experimental_measurements=m, sigmas=sigmas)
    amm_convenience = amm_convenience_estimator.fit(counts).fetch_model()
    assert_equal(tau, amm.lagtime)
    assert_array_almost_equal(E, amm_estimator.expectations_by_state)
    assert_array_almost_equal(E, amm_convenience_estimator.expectations_by_state, decimal=4)
    assert_array_almost_equal(m, amm_estimator.experimental_measurements)
    assert_array_almost_equal(m, amm_convenience_estimator.experimental_measurements)
    assert_array_almost_equal(w, amm_estimator.experimental_measurement_weights)
    assert_array_almost_equal(w, amm_convenience_estimator.experimental_measurement_weights)
    assert_array_almost_equal(amm.transition_matrix, amm_convenience.transition_matrix, decimal=4)
    assert_array_almost_equal(amm.stationary_distribution, amm_convenience.stationary_distribution, decimal=4)
    assert_array_almost_equal(amm.optimizer_state.lagrange, amm_convenience.optimizer_state.lagrange, decimal=4)
