import pytest
import numpy as np
import scipy.sparse as sparse
from numpy.testing import assert_equal, assert_array_almost_equal, assert_almost_equal

from deeptime.data import birth_death_chain
from deeptime.markov.msm import MarkovStateModel
from deeptime.markov.tools.analysis import stationary_distribution
from deeptime.markov.tools.flux import flux_production, flux_producers, flux_consumers, total_flux


def _to_dense(arr):
    if sparse.issparse(arr):
        return arr.toarray()
    return arr


@pytest.fixture
def tpt_scenario(sparse_mode):
    P = np.array([[0.8, 0.15, 0.05, 0.0, 0.0],
                  [0.1, 0.75, 0.05, 0.05, 0.05],
                  [0.05, 0.1, 0.8, 0.0, 0.05],
                  [0.0, 0.2, 0.0, 0.8, 0.0],
                  [0.0, 0.02, 0.02, 0.0, 0.96]])
    if sparse_mode:
        P = sparse.csr_matrix(P)
    msm = MarkovStateModel(P)
    tpt = msm.reactive_flux([0], [4])
    return msm, tpt


@pytest.fixture
def tpt_scenario_bd(sparse_mode):
    p = np.zeros(10)
    q = np.zeros(10)
    p[0:-1] = 0.5
    q[1:] = 0.5
    p[4] = 0.01
    q[6] = 0.1

    bdc = birth_death_chain(q, p, sparse=sparse_mode)
    tpt = bdc.msm.reactive_flux([0, 1], [8, 9])
    return tpt, bdc


def test_bd_netflux(tpt_scenario_bd):
    assert_array_almost_equal(_to_dense(tpt_scenario_bd[0].net_flux), _to_dense(tpt_scenario_bd[1].netflux(1, 8)))


def test_bd_flux(tpt_scenario_bd):
    assert_array_almost_equal(_to_dense(tpt_scenario_bd[0].gross_flux), _to_dense(tpt_scenario_bd[1].flux(1, 8)))


def test_bd_totalflux(tpt_scenario_bd):
    assert_array_almost_equal(_to_dense(tpt_scenario_bd[0].total_flux), _to_dense(tpt_scenario_bd[1].totalflux(1, 8)))


def test_bd_rate(tpt_scenario_bd):
    assert_almost_equal(tpt_scenario_bd[0].rate, tpt_scenario_bd[1].rate(1, 8))


def test_nstates(tpt_scenario):
    msm, tpt = tpt_scenario
    assert_equal(msm.n_states, tpt.n_states)


def test_source_states(tpt_scenario):
    assert_equal(tpt_scenario[1].source_states, [0])


def test_intermediate_states(tpt_scenario):
    assert_equal(tpt_scenario[1].intermediate_states, [1, 2, 3])


def test_target_states(tpt_scenario):
    assert_equal(tpt_scenario[1].target_states, [4])


def test_netflux(tpt_scenario):
    flux = _to_dense(tpt_scenario[1].net_flux)
    assert_array_almost_equal(
        flux,
        np.array([[0.00000000e+00, 7.71791768e-03, 3.08716707e-03, 0.00000000e+00, 0.00000000e+00],
                  [0.00000000e+00, 0.00000000e+00, 5.14527845e-04, 0.00000000e+00, 7.20338983e-03],
                  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.60169492e-03],
                  [0.00000000e+00, 4.33680869e-19, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]])
    )


def test_flux_production(tpt_scenario):
    prod = flux_production(tpt_scenario[1].net_flux)
    assert_array_almost_equal(prod, [1.080508e-2, 0, 0, 0, -1.080508e-2])


def test_flux_producers(tpt_scenario):
    producers = flux_producers(tpt_scenario[1].net_flux)
    assert_equal(producers, [0])


def test_flux_consumers(tpt_scenario):
    consumers = flux_consumers(tpt_scenario[1].net_flux)
    assert_equal(consumers, [4])


def test_grossflux(tpt_scenario):
    grossflux = _to_dense(tpt_scenario[1].gross_flux)
    assert_array_almost_equal(
        grossflux,
        np.array([[0., 0.00771792, 0.00308717, 0., 0.],
                  [0., 0., 0.00308717, 0.00257264, 0.00720339],
                  [0., 0.00257264, 0., 0., 0.00360169],
                  [0., 0.00257264, 0., 0., 0.],
                  [0., 0., 0., 0., 0.]])
    )


def test_total_flux(tpt_scenario):
    assert_almost_equal(tpt_scenario[1].total_flux, 0.0108050847458)
    assert_almost_equal(total_flux(tpt_scenario[1].net_flux), 0.0108050847458)
    assert_almost_equal(total_flux(tpt_scenario[1].net_flux, A=[0]), 0.0108050847458)


def test_forward_committor(tpt_scenario):
    assert_array_almost_equal(tpt_scenario[1].forward_committor, np.array([0., 0.35714286, 0.42857143, 0.35714286, 1.]))


def test_backward_committor(tpt_scenario):
    assert_array_almost_equal(tpt_scenario[1].backward_committor, np.array([1., 0.65384615, 0.53125, 0.65384615, 0.]))


def test_rate(tpt_scenario):
    assert_almost_equal(tpt_scenario[1].rate, 0.0272727272727)
    assert_almost_equal(tpt_scenario[1].mfpt, 36.6666666667)


def test_pathways(tpt_scenario):
    ref_paths = [[0, 1, 4], [0, 2, 4], [0, 1, 2, 4]]
    ref_pathfluxes = np.array([0.00720338983051, 0.00308716707022, 0.000514527845036])

    ref_paths_95percent = [[0, 1, 4], [0, 2, 4]]
    ref_pathfluxes_95percent = np.array([0.00720338983051, 0.00308716707022])

    tpt = tpt_scenario[1]
    # all paths
    paths, pathfluxes = tpt.pathways()
    assert_equal(len(paths), len(ref_paths))
    for i in range(len(paths)):
        assert_equal(paths[i], ref_paths[i])

    assert_almost_equal(pathfluxes, ref_pathfluxes)
    # major paths
    paths, pathfluxes = tpt.pathways(fraction=0.95)
    assert_equal(len(paths), len(ref_paths_95percent))
    for i in range(len(paths)):
        assert_equal(paths[i], ref_paths_95percent[i])
    assert_almost_equal(pathfluxes, ref_pathfluxes_95percent)


@pytest.mark.parametrize("fraction,ref", [
    (1.0, np.array([[0.00000000e+00, 7.71791768e-03, 3.08716707e-03, 0.00000000e+00, 0.00000000e+00],
                    [0.00000000e+00, 0.00000000e+00, 5.14527845e-04, 0.00000000e+00, 7.20338983e-03],
                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.60169492e-03],
                    [0.00000000e+00, 4.33680869e-19, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]])),
    (0.95, np.array([[0., 0.00720339, 0.00308717, 0., 0.],
                     [0., 0., 0., 0., 0.00720339],
                     [0., 0., 0., 0., 0.00308717],
                     [0., 0., 0., 0., 0.],
                     [0., 0., 0., 0., 0.]]))
])
def test_major_flux(tpt_scenario, fraction, ref):
    flux = _to_dense(tpt_scenario[1].major_flux(fraction))
    assert_array_almost_equal(flux, ref)


def test_coarse_grain(sparse_mode):
    # 16-state toy system
    P_nonrev = np.array([[0.5, 0.2, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.2, 0.5, 0.1, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.1, 0.5, 0.2, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.1, 0.5, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.3, 0.0, 0.0, 0.0, 0.5, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.1, 0.0, 0.0, 0.2, 0.5, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.5, 0.2, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.3, 0.5, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.5, 0.1, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.2, 0.5, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.5, 0.1, 0.0, 0.0, 0.2, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.2, 0.5, 0.0, 0.0, 0.0, 0.2],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.5, 0.2, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.3, 0.5, 0.1, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.1, 0.5, 0.2],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.2, 0.5]])
    pstat2_nonrev = stationary_distribution(P_nonrev)
    # make reversible
    C = np.dot(np.diag(pstat2_nonrev), P_nonrev)
    Csym = C + C.T
    P = Csym / np.sum(Csym, axis=1)[:, np.newaxis]
    if sparse_mode:
        P = sparse.csr_matrix(P)
    msm = MarkovStateModel(P)
    tpt = msm.reactive_flux([0, 4], [11, 15])
    coarse_sets = [[2, 3, 6, 7], [10, 11, 14, 15], [0, 1, 4, 5], [8, 9, 12, 13], ]
    tpt_sets, cgRF = tpt.coarse_grain(coarse_sets)
    assert_equal(tpt_sets, [{0, 4}, {2, 3, 6, 7}, {10, 14}, {1, 5}, {8, 9, 12, 13}, {11, 15}])
    assert_equal(cgRF.source_states, [0])
    assert_equal(cgRF.intermediate_states, [1, 2, 3, 4])
    assert_equal(cgRF.target_states, [5])
    assert_array_almost_equal(cgRF.stationary_distribution,
                              np.array([0.15995388, 0.18360442, 0.12990937, 0.11002342, 0.31928127, 0.09722765]))
    assert_array_almost_equal(cgRF.forward_committor,
                              np.array([0., 0.56060272, 0.73052426, 0.19770537, 0.36514272, 1.]))
    assert_array_almost_equal(cgRF.backward_committor,
                              np.array([1., 0.43939728, 0.26947574, 0.80229463, 0.63485728, 0.]))
    assert_array_almost_equal(_to_dense(cgRF.net_flux),
                              np.array([[0., 0., 0., 0.00427986, 0.00282259, 0.],
                                        [0., 0., 0.00120686, 0., 0., 0.00201899],
                                        [0., 0., 0., 0., 0., 0.00508346],
                                        [0., 0.00322585, 0., 0., 0.00105401, 0.],
                                        [0., 0., 0.0038766, 0., 0., 0.],
                                        [0., 0., 0., 0., 0., 0.]]))
    assert_array_almost_equal(_to_dense(cgRF.gross_flux),
                              np.array([[0., 0., 0., 0.00427986, 0.00282259, 0.],
                                        [0., 0, 0.00234578, 0.00104307, 0., 0.00201899],
                                        [0., 0.00113892, 0, 0., 0.00142583, 0.00508346],
                                        [0., 0.00426892, 0., 0, 0.00190226, 0.],
                                        [0., 0., 0.00530243, 0.00084825, 0, 0.],
                                        [0., 0., 0., 0., 0., 0.]]),
                              decimal=6)
