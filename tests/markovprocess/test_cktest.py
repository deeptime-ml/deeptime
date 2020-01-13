# This file is part of PyEMMA.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# PyEMMA is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


r"""Unit test for Chapman-Kolmogorov-Test module

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""

import unittest

import numpy as np
from msmtools.estimation import count_matrix, largest_connected_set, largest_connected_submatrix, transition_matrix
from msmtools.generation import generate_traj
from msmtools.util.birth_death_chain import BirthDeathChain

from sktime.markovprocess.maximum_likelihood_hmsm import MaximumLikelihoodHMSM
from sktime.datasets import double_well_discrete
from sktime.lagged_model_validator import LaggedModelValidation
from sktime.markovprocess import cktest
from sktime.markovprocess.bayesian_hmsm import BayesianHMSM
from tests.markovprocess.factory import bayesian_markov_model
from tests.markovprocess.test_hmsm import estimate_hidden_markov_model
from tests.markovprocess.test_msm import estimate_markov_model


class TestCK_MSM(unittest.TestCase):
    def setUp(self):
        """Store state of the rng"""
        self.state = np.random.mtrand.get_state()

        """Reseed the rng to enforce 'deterministic' behavior"""
        np.random.mtrand.seed(42)

        """Meta-stable birth-death chain"""
        b = 2
        q = np.zeros(7)
        p = np.zeros(7)
        q[1:] = 0.5
        p[0:-1] = 0.5
        q[2] = 1.0 - 10 ** (-b)
        q[4] = 10 ** (-b)
        p[2] = 10 ** (-b)
        p[4] = 1.0 - 10 ** (-b)

        bdc = BirthDeathChain(q, p)
        P = bdc.transition_matrix()
        dtraj = generate_traj(P, 10000, start=0)
        tau = 1

        """Estimate MSM"""
        estimator, MSM = estimate_markov_model(dtraj, tau, return_estimator=True)
        self.estimator = estimator
        P_MSM = MSM.transition_matrix
        mu_MSM = MSM.stationary_distribution

        """Meta-stable sets"""
        A = [0, 1, 2]
        B = [4, 5, 6]

        w_MSM = np.zeros((2, mu_MSM.shape[0]))
        w_MSM[0, A] = mu_MSM[A] / mu_MSM[A].sum()
        w_MSM[1, B] = mu_MSM[B] / mu_MSM[B].sum()

        K = 10
        P_MSM_dense = P_MSM

        p_MSM = np.zeros((K, 2))
        w_MSM_k = 1.0 * w_MSM
        for k in range(1, K):
            w_MSM_k = np.dot(w_MSM_k, P_MSM_dense)
            p_MSM[k, 0] = w_MSM_k[0, A].sum()
            p_MSM[k, 1] = w_MSM_k[1, B].sum()

        """Assume that sets are equal, A(\tau)=A(k \tau) for all k"""
        w_MD = 1.0 * w_MSM
        p_MD = np.zeros((K, 2))
        eps_MD = np.zeros((K, 2))

        for k in range(1, K):
            """Build MSM at lagtime k*tau"""
            C_MD = count_matrix(dtraj, k * tau, sliding=True) / (k * tau)
            lcc_MD = largest_connected_set(C_MD)
            Ccc_MD = largest_connected_submatrix(C_MD, lcc=lcc_MD)
            c_MD = Ccc_MD.sum(axis=1)
            P_MD = transition_matrix(Ccc_MD).toarray()
            w_MD_k = np.dot(w_MD, P_MD)

            """Set A"""
            prob_MD = w_MD_k[0, A].sum()
            c = c_MD[A].sum()
            p_MD[k, 0] = prob_MD
            eps_MD[k, 0] = np.sqrt(k * (prob_MD - prob_MD ** 2) / c)

            """Set B"""
            prob_MD = w_MD_k[1, B].sum()
            c = c_MD[B].sum()
            p_MD[k, 1] = prob_MD
            eps_MD[k, 1] = np.sqrt(k * (prob_MD - prob_MD ** 2) / c)

        """Input"""
        self.MSM = MSM
        self.K = K
        self.A = A
        self.B = B

        """Expected results"""
        # skip first result as it is trivial case of mlag=0
        self.p_MSM = p_MSM[1:, :]
        self.p_MD = p_MD[1:, :]
        self.eps_MD = eps_MD[1:, :]

        self.dtraj = dtraj

    def tearDown(self):
        """Revert the state of the rng"""
        np.random.mtrand.set_state(self.state)

    def test_cktest(self):
        # introduce a (fake) third set in order to model incomplete partition.
        memberships = np.array([[1, 0, 0],
                                [1, 0, 0],
                                [1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1],
                                [0, 0, 1],
                                [0, 0, 1]])
        ck = cktest(test_model=self.MSM, test_estimator=self.estimator, dtrajs=self.dtraj, nsets=3,
                    memberships=memberships)
        ck = ck.fetch_model()
        p_MSM = np.vstack([ck.predictions[:, 0, 0], ck.predictions[:, 2, 2]]).T
        np.testing.assert_allclose(p_MSM, self.p_MSM)
        p_MD = np.vstack([ck.estimates[:, 0, 0], ck.estimates[:, 2, 2]]).T
        np.testing.assert_allclose(p_MD, self.p_MD, rtol=1e-5, atol=1e-8)


class TestCK_AllEstimators(unittest.TestCase):
    """ Integration tests for various estimators"""

    def test_ck_msm(self):
        estimator, MLMSM = estimate_markov_model([double_well_discrete().dtraj_n6good], 40,
                                                 return_estimator=True)
        with self.assertRaises(ValueError):
            cktest(estimator, MLMSM, nsets=2, mlags=[0, 1, 50], dtrajs=double_well_discrete().dtraj_n6good)

        self.ck = cktest(test_estimator=estimator, test_model=MLMSM, nsets=2, mlags=[1, 10],
                         dtrajs=double_well_discrete().dtraj_n6good).fetch_model()
        assert isinstance(self.ck, LaggedModelValidation)
        estref = np.array([[[0.89806859, 0.10193141],
                            [0.10003466, 0.89996534]],
                           [[0.64851782, 0.35148218],
                            [0.34411751, 0.65588249]]])
        predref = np.array([[[0.89806859, 0.10193141],
                             [0.10003466, 0.89996534]],
                            [[0.62613723, 0.37386277],
                             [0.3669059, 0.6330941]]])
        # rough agreement with MLE
        np.testing.assert_allclose(self.ck.estimates, estref, rtol=0.1, atol=10.0)
        assert self.ck.estimates_conf[0] is None
        assert self.ck.estimates_conf[1] is None
        np.testing.assert_allclose(self.ck.predictions, predref, rtol=0.1, atol=10.0)
        assert self.ck.predictions_conf[0] is None
        assert self.ck.predictions_conf[1] is None

    def test_its_bmsm(self):
        estimator, BMSM = bayesian_markov_model(double_well_discrete().dtraj_n6good, 40, reversible=True,
                                                return_estimator=True)
        # also ensure that reversible bit does not flip during cktest
        assert BMSM.prior.is_reversible
        self.ck = cktest(test_estimator=estimator, test_model=BMSM.prior, dtrajs=double_well_discrete().dtraj_n6good,
                         nsets=2, mlags=[1, 10]).fetch_model()
        assert isinstance(self.ck, LaggedModelValidation)
        assert BMSM.prior.is_reversible
        estref = np.array([
                           [[0.89722931, 0.10277069],
                            [0.10070029, 0.89929971]],
                           [[0.64668027, 0.35331973],
                            [0.34369109, 0.65630891]]])
        predref = np.array([
                            [[0.89722931, 0.10277069],
                             [0.10070029, 0.89929971]],
                            [[0.62568693, 0.37431307],
                             [0.36677222, 0.63322778]]])
        predLref = np.array([
                             [[0.89398296, 0.09942586],
                              [0.09746008, 0.89588256]],
                             [[0.6074675, 0.35695492],
                              [0.34831224, 0.61440531]]])
        predRref = np.array([
                             [[0.90070139, 0.10630301],
                              [0.10456111, 0.90255169]],
                             [[0.64392557, 0.39258944],
                              [0.38762444, 0.65176265]]])
        # rough agreement
        assert np.allclose(self.ck.estimates, estref, rtol=0.1, atol=10.0)
        assert self.ck.estimates_conf[0] is None
        assert self.ck.estimates_conf[1] is None
        assert np.allclose(self.ck.predictions, predref, rtol=0.1, atol=10.0)
        assert np.allclose(self.ck.predictions[0], predLref, rtol=0.1, atol=10.0)
        assert np.allclose(self.ck.predictions[1], predRref, rtol=0.1, atol=10.0)

    def test_its_hmsm(self):
        dtraj = [double_well_discrete().dtraj_n6good]
        est = MaximumLikelihoodHMSM(n_states=2, lagtime=10)
        MLHMM = est.fit(dtraj).fetch_model()
        states = MLHMM.states_largest()
        obs = MLHMM.nonempty_obs(dtraj)
        cktest(est, MLHMM, dtraj, )


        # est, MLHMM = estimate_hidden_markov_model(dtraj, 2, 10, return_estimator=True)
        self.ck = cktest(test_estimator=est, test_model=MLHMM, dtrajs=dtraj, mlags=[1, 10], nsets=2).fetch_model()
        estref = np.array([
                           [[0.98515058, 0.01484942],
                            [0.01442843, 0.98557157]],
                           [[0.88172685, 0.11827315],
                            [0.11878823, 0.88121177]]])
        predref = np.array([
                            [[0.98515058, 0.01484942],
                             [0.01442843, 0.98557157]],
                            [[0.86961812, 0.13038188],
                             [0.12668553, 0.87331447]]])
        # rough agreement with MLE
        assert np.allclose(self.ck.estimates, estref, rtol=0.1, atol=10.0)
        assert self.ck.estimates_conf[0] is None
        assert self.ck.estimates_conf[1] is None
        assert np.allclose(self.ck.predictions, predref, rtol=0.1, atol=10.0)
        assert self.ck.predictions_conf[0] is None
        assert self.ck.predictions_conf[1] is None

    def test_its_bhmm(self):
        dtraj = double_well_discrete().dtraj_n6good
        bhmm = BayesianHMSM.default(dtraj, n_states=2, lagtime=10).fit(dtrajs=dtraj)
        self.ck = bhmm.cktest(dtraj, mlags=[1, 10])
        estref = np.array([
                           [[0.98497185, 0.01502815],
                            [0.01459256, 0.98540744]],
                           [[0.88213404, 0.11786596],
                            [0.11877379, 0.88122621]]])
        predref = np.array([
                            [[0.98497185, 0.01502815],
                             [0.01459256, 0.98540744]],
                            [[0.86824695, 0.13175305],
                             [0.1279342, 0.8720658]]])
        predLref = np.array([
                             [[0.98282734, 0.01284444],
                              [0.0123793, 0.98296742]],
                             [[0.8514399, 0.11369687],
                              [0.10984971, 0.85255827]]])
        predRref = np.array([
                             [[0.98715575, 0.01722138],
                              [0.0178059, 0.98762081]],
                             [[0.8865478, 0.14905352],
                              [0.14860461, 0.89064809]]])
        # rough agreement
        assert np.allclose(self.ck.estimates, estref, rtol=0.1, atol=10.0)
        assert self.ck.estimates_conf[0] is None
        assert self.ck.estimates_conf[1] is None
        assert np.allclose(self.ck.predictions, predref, rtol=0.1, atol=10.0)
        assert np.allclose(self.ck.predictions[0], predLref, rtol=0.1, atol=10.0)
        assert np.allclose(self.ck.predictions[1], predRref, rtol=0.1, atol=10.0)

if __name__ == "__main__":
    unittest.main()
