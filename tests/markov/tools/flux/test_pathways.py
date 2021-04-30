"""Unit test for the reaction pathway decomposition

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>
.. moduleauthor:: marscher
.. moduleauthor:: clonker

"""

import numpy as np
import pytest
from numpy.testing import assert_equal
from scipy.sparse import csr_matrix

from deeptime.markov.tools.flux import pathways
from tests.markov.tools.numeric import assert_allclose


@pytest.fixture
def scenario(sparse_mode):
    """Small flux-network"""
    flux_matrix = np.zeros((8, 8))
    flux_matrix[0, 2] = 10.0
    flux_matrix[2, 6] = 10.0
    flux_matrix[1, 3] = 100.0
    flux_matrix[3, 4] = 30.0
    flux_matrix[3, 5] = 70.0
    flux_matrix[4, 6] = 5.0
    flux_matrix[4, 7] = 25.0
    flux_matrix[5, 6] = 30.0
    flux_matrix[5, 7] = 40.0
    if sparse_mode:
        flux_matrix = csr_matrix(flux_matrix)
    """Reactant and product states"""
    A = [0, 1]
    B = [6, 7]
    paths = [
        np.array([1, 3, 5, 7]),
        np.array([1, 3, 5, 6]),
        np.array([1, 3, 4, 7]),
        np.array([0, 2, 6]),
        np.array([1, 3, 4, 6])
    ]
    capacities = [40., 30., 25., 10., 5.]
    return flux_matrix, paths, capacities, A, B


def test_pathways(scenario):
    flux_matrix, ref_paths, ref_capacities, source, target = scenario
    paths, capacities = pathways(flux_matrix, source, target)
    assert_equal(len(paths), len(ref_paths))
    assert_equal(len(capacities), len(ref_capacities))

    for i in range(len(paths)):
        assert_equal(paths[i], ref_paths[i])
        assert_equal(capacities[i], ref_capacities[i])


def test_pathways_incomplete(scenario):
    flux_matrix, ref_paths, ref_capacities, source, target = scenario
    paths, capacities = pathways(flux_matrix, source, target, fraction=.5)
    assert_equal(len(paths), 2)
    assert_equal(len(capacities), 2)

    for i in range(len(paths)):
        assert_equal(paths[i], ref_paths[i])
        assert_allclose(capacities[i], ref_capacities[i])

    with pytest.warns(RuntimeWarning):
        paths, capacities = pathways(flux_matrix, [0, 1], [6, 7], fraction=1.0, maxiter=1)
        for i in range(len(paths)):
            assert_allclose(paths[i], ref_paths[i])
            assert_allclose(capacities[i], ref_capacities[i])


def test_with_almost_converged_stat_dist(sparse_mode):
    """ test for https://github.com/markovmodel/msmtools/issues/106 """
    from deeptime.markov.tools.analysis import committor, is_reversible
    from deeptime.markov.tools.flux import flux_matrix, to_netflux
    from deeptime.markov import reactive_flux, ReactiveFlux

    T = np.array([[0.2576419223095193, 0.2254214623509954, 0.248270708174756,
                   0.2686659071647294],
                  [0.2233847186210225, 0.2130434781715344, 0.2793477268264001,
                   0.284224076381043],
                  [0.2118717275169231, 0.2405661227681972, 0.2943396213976011,
                   0.2532225283172787],
                  [0.2328617711043517, 0.2485926610067547, 0.2571819311236834,
                   0.2613636367652102]])
    if sparse_mode:
        T = csr_matrix(T)
    mu = np.array([0.2306979668517676, 0.2328013892993006, 0.2703312416016573,
                   0.2661694022472743])
    assert is_reversible(T)
    np.testing.assert_allclose(T.T.dot(mu).T, mu)
    np.testing.assert_equal(T.T.dot(mu).T, T.T.dot(mu))
    A = [0]
    B = [1]

    # forward committor
    qplus = committor(T, A, B, forward=True, mu=mu)
    # backward committor
    if is_reversible(T, mu=mu):
        qminus = 1.0 - qplus
    else:
        qminus = committor(T, A, B, forward=False, mu=mu)

    tpt_obj = reactive_flux(T, A, B)
    tpt_obj.major_flux(1.0)
    # gross flux
    grossflux = flux_matrix(T, mu, qminus, qplus, netflux=False)
    # net flux
    netflux = to_netflux(grossflux)

    F = ReactiveFlux(A, B, netflux, stationary_distribution=mu, qminus=qminus, qplus=qplus, gross_flux=grossflux)
    F.pathways(1.0)
