r"""Unit test for the ReactiveFlux object

.. moduleauthor:: F.Noe <frank  DOT noe AT fu-berlin DOT de>

"""
import unittest

import deeptime.markov.tools.analysis as msmana
import numpy as np
from numpy.testing import assert_allclose

from deeptime.markov import TransitionCountModel, reactive_flux
from deeptime.markov.msm import MarkovStateModel


class TestReactiveFluxFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 5-state toy system
        cls.P = MarkovStateModel(
            np.array([[0.8, 0.15, 0.05, 0.0, 0.0],
                      [0.1, 0.75, 0.05, 0.05, 0.05],
                      [0.05, 0.1, 0.8, 0.0, 0.05],
                      [0.0, 0.2, 0.0, 0.8, 0.0],
                      [0.0, 0.02, 0.02, 0.0, 0.96]]))
        cls.A = [0]
        cls.B = [4]
        cls.I = [1, 2, 3]

        # REFERENCE SOLUTION FOR PATH DECOMP
        cls.ref_committor = np.array([0., 0.35714286, 0.42857143, 0.35714286, 1.])
        cls.ref_backwardcommittor = np.array([1., 0.65384615, 0.53125, 0.65384615, 0.])
        cls.ref_grossflux = np.array([[0., 0.00771792, 0.00308717, 0., 0.],
                                      [0., 0., 0.00308717, 0.00257264, 0.00720339],
                                      [0., 0.00257264, 0., 0., 0.00360169],
                                      [0., 0.00257264, 0., 0., 0.],
                                      [0., 0., 0., 0., 0.]])
        cls.ref_netflux = np.array([[0.00000000e+00, 7.71791768e-03, 3.08716707e-03, 0.00000000e+00, 0.00000000e+00],
                                    [0.00000000e+00, 0.00000000e+00, 5.14527845e-04, 0.00000000e+00, 7.20338983e-03],
                                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.60169492e-03],
                                    [0.00000000e+00, 4.33680869e-19, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]])
        cls.ref_totalflux = 0.0108050847458
        cls.ref_kAB = 0.0272727272727
        cls.ref_mfptAB = 36.6666666667

        cls.ref_paths = [[0, 1, 4], [0, 2, 4], [0, 1, 2, 4]]
        cls.ref_pathfluxes = np.array([0.00720338983051, 0.00308716707022, 0.000514527845036])

        cls.ref_paths_95percent = [[0, 1, 4], [0, 2, 4]]
        cls.ref_pathfluxes_95percent = np.array([0.00720338983051, 0.00308716707022])
        cls.ref_majorflux_95percent = np.array([[0., 0.00720339, 0.00308717, 0., 0.],
                                                [0., 0., 0., 0., 0.00720339],
                                                [0., 0., 0., 0., 0.00308717],
                                                [0., 0., 0., 0., 0.],
                                                [0., 0., 0., 0., 0.]])

        # Testing:
        cls.tpt1 = cls.P.reactive_flux(cls.A, cls.B)

        # 16-state toy system
        P2_nonrev = np.array([[0.5, 0.2, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
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
        pstat2_nonrev = msmana.stationary_distribution(P2_nonrev)
        # make reversible
        C = np.dot(np.diag(pstat2_nonrev), P2_nonrev)
        Csym = C + C.T
        cls.P2 = MarkovStateModel(Csym / np.sum(Csym, axis=1)[:, np.newaxis])
        cls.A2 = [0, 4]
        cls.B2 = [11, 15]
        cls.coarsesets2 = [[2, 3, 6, 7], [10, 11, 14, 15], [0, 1, 4, 5], [8, 9, 12, 13], ]

        # REFERENCE SOLUTION CG
        cls.ref2_tpt_sets = [{0, 4}, {2, 3, 6, 7}, {10, 14}, {1, 5}, {8, 9, 12, 13},
                             {11, 15}]
        cls.ref2_cgA = [0]
        cls.ref2_cgI = [1, 2, 3, 4]
        cls.ref2_cgB = [5]
        cls.ref2_cgpstat = np.array([0.15995388, 0.18360442, 0.12990937, 0.11002342, 0.31928127, 0.09722765])
        cls.ref2_cgcommittor = np.array([0., 0.56060272, 0.73052426, 0.19770537, 0.36514272, 1.])
        cls.ref2_cgbackwardcommittor = np.array([1., 0.43939728, 0.26947574, 0.80229463, 0.63485728, 0.])
        cls.ref2_cggrossflux = np.array([[0., 0., 0., 0.00427986, 0.00282259, 0.],
                                         [0., 0, 0.00234578, 0.00104307, 0., 0.00201899],
                                         [0., 0.00113892, 0, 0., 0.00142583, 0.00508346],
                                         [0., 0.00426892, 0., 0, 0.00190226, 0.],
                                         [0., 0., 0.00530243, 0.00084825, 0, 0.],
                                         [0., 0., 0., 0., 0., 0.]])
        cls.ref2_cgnetflux = np.array([[0., 0., 0., 0.00427986, 0.00282259, 0.],
                                       [0., 0., 0.00120686, 0., 0., 0.00201899],
                                       [0., 0., 0., 0., 0., 0.00508346],
                                       [0., 0.00322585, 0., 0., 0.00105401, 0.],
                                       [0., 0., 0.0038766, 0., 0., 0.],
                                       [0., 0., 0., 0., 0., 0.]])

        # Testing
        cls.tpt2 = cls.P2.reactive_flux(cls.A2, cls.B2)

    def test_invalid_inputs(self):
        with np.testing.assert_raises(ValueError):
            self.P.reactive_flux(source_states=[], target_states=[1])
        with np.testing.assert_raises(ValueError):
            self.P.reactive_flux(source_states=[1], target_states=[])
        with np.testing.assert_raises(ValueError):
            self.P.reactive_flux(source_states=np.arange(self.P.n_states + 1), target_states=[0])
        with np.testing.assert_raises(ValueError):
            self.P.reactive_flux(source_states=[0], target_states=np.arange(self.P.n_states + 1))
        with np.testing.assert_raises(ValueError):
            reactive_flux(np.array([[.5, .6], [.3, .7]]), [0], [1])  # not a transition matrix

    def test_n_states(self):
        self.assertEqual(self.tpt1.n_states, self.P.n_states)

    def test_A(self):
        self.assertEqual(self.tpt1.source_states, self.A)

    def test_I(self):
        self.assertEqual(self.tpt1.intermediate_states, self.I)

    def test_B(self):
        self.assertEqual(self.tpt1.target_states, self.B)

    def test_netflux(self):
        assert_allclose(self.tpt1.net_flux, self.ref_netflux, rtol=1e-02, atol=1e-07)

    def test_grossflux(self):
        assert_allclose(self.tpt1.gross_flux, self.ref_grossflux, rtol=1e-02, atol=1e-07)

    def test_forwardcommittor(self):
        assert_allclose(self.tpt1.forward_committor, self.ref_committor, rtol=1e-02, atol=1e-07)

    def test_backwardcommittor(self):
        assert_allclose(self.tpt1.backward_committor, self.ref_backwardcommittor, rtol=1e-02, atol=1e-07)

    def test_total_flux(self):
        assert_allclose(self.tpt1.total_flux, self.ref_totalflux, rtol=1e-02, atol=1e-07)

    def test_rate(self):
        assert_allclose(self.tpt1.rate, self.ref_kAB, rtol=1e-02, atol=1e-07)
        assert_allclose(1.0 / self.tpt1.rate, self.ref_mfptAB, rtol=1e-02, atol=1e-07)
        assert_allclose(self.tpt1.mfpt, self.ref_mfptAB, rtol=1e-02, atol=1e-07)

    """These tests are broken since the matrix is irreversible and no
    pathway-decomposition can be computed for irreversible matrices"""

    def test_pathways(self):
        # all paths
        paths, pathfluxes = self.tpt1.pathways()
        self.assertEqual(len(paths), len(self.ref_paths))
        for i in range(len(paths)):
            self.assertTrue(np.all(np.array(paths[i]) == np.array(self.ref_paths[i])))
        assert_allclose(pathfluxes, self.ref_pathfluxes, rtol=1e-02, atol=1e-07)
        # major paths
        paths, pathfluxes = self.tpt1.pathways(fraction=0.95)
        self.assertEqual(len(paths), len(self.ref_paths_95percent))
        for i in range(len(paths)):
            self.assertTrue(np.all(np.array(paths[i]) == np.array(self.ref_paths_95percent[i])))
        assert_allclose(pathfluxes, self.ref_pathfluxes_95percent, rtol=1e-02, atol=1e-07)

    def test_major_flux(self):
        # all flux
        assert_allclose(self.tpt1.major_flux(fraction=1.0), self.ref_netflux, rtol=1e-02, atol=1e-07)
        # 0.99 flux
        assert_allclose(self.tpt1.major_flux(fraction=0.95), self.ref_majorflux_95percent, rtol=1e-02, atol=1e-07)

    def test_dt_model(self):
        C = TransitionCountModel(np.array([[0.1, 0.9], [0.9, 0.1]]), lagtime=5)
        msm = MarkovStateModel(C.count_matrix, count_model=C)
        tpt = msm.reactive_flux([0], [1])
        np.testing.assert_equal(msm.lagtime, 5)

    def test_coarse_grain(self):
        tpt_sets, cg_flux = self.tpt2.coarse_grain(self.coarsesets2)
        self.assertEqual(tpt_sets, self.ref2_tpt_sets)
        self.assertEqual(cg_flux.source_states, self.ref2_cgA)
        self.assertEqual(cg_flux.intermediate_states, self.ref2_cgI)
        self.assertEqual(cg_flux.target_states, self.ref2_cgB)
        assert_allclose(cg_flux.stationary_distribution, self.ref2_cgpstat)
        assert_allclose(cg_flux.forward_committor, self.ref2_cgcommittor)
        assert_allclose(cg_flux.backward_committor, self.ref2_cgbackwardcommittor)
        assert_allclose(cg_flux.net_flux, self.ref2_cgnetflux, rtol=1.e-5, atol=1.e-8)
        assert_allclose(cg_flux.gross_flux, self.ref2_cggrossflux, rtol=1.e-5, atol=1.e-8)
