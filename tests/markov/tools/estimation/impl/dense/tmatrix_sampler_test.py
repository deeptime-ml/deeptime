# This file is part of scikit-time and MSMTools.
#
# Copyright (c) 2020, 2015, 2014 AI4Science Group, Freie Universitaet Berlin (GER)
#
# scikit-time and MSMTools is free software: you can redistribute it and/or modify
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

r"""Unit tests for the covariance module

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""

import numpy as np
import pytest
from sktime.markov.tools.estimation import transition_matrix as tmatrix
from sktime.markov.tools.estimation.dense.tmat_sampling.tmatrix_sampler import TransitionMatrixSampler


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_non_reversible(dtype):
    C = np.array([[7048, 6, 2], [6, 2, 3], [2, 3, 2933]], dtype=dtype)

    # Mean in the asymptotic limit, N_samples -> \infty
    alpha = C.astype(np.float64)
    alpha0 = alpha.sum(axis=1)

    mean = alpha / alpha0[:, np.newaxis]
    var = alpha * (alpha0[:, np.newaxis] - alpha) / (alpha0 ** 2 * (alpha0 + 1.0))[:, np.newaxis]

    N = 1000
    """Create sampler object"""
    sampler = TransitionMatrixSampler(C, reversible=False, seed=42)

    # Compute sample mean
    sampled_mean = np.zeros(C.shape)
    for i in range(N):
        s = sampler.sample()
        np.testing.assert_equal(s.dtype, dtype)
        sampled_mean += s
    sampled_mean *= 1.0 / N

    # Check if sample mean and true mean approximately fall into the 2\sigma interval
    max_deviation = np.max(np.abs(np.abs(sampled_mean - mean) - 2.0 * np.sqrt(var / N)))
    np.testing.assert_(max_deviation < 0.01)


@pytest.mark.parametrize("dtype", (np.float32, np.float64, np.longdouble))
def test_reversible(dtype):
    C = 1.0 * np.array([[7048, 6, 0], [6, 2, 3], [0, 3, 2933]]).astype(dtype)
    P_mle = tmatrix(C, reversible=True)
    N = 1000

    sampler = TransitionMatrixSampler(C, reversible=True)

    sample = np.zeros((N, 3, 3))
    for i in range(N):
        s = sampler.sample()
        np.testing.assert_equal(s.dtype, dtype)
        sample[i, :, :] = s
    mean = np.mean(sample, axis=0)
    std = np.std(sample, axis=0)

    # Check if sample mean and MLE agree within the sample standard deviation
    diff = np.abs(mean - P_mle)
    np.testing.assert_(np.all(diff <= std))


@pytest.mark.parametrize("dtype", (np.float32, np.float64, np.longdouble))
def test_reversible_pi(dtype):
    C = np.array([[7048, 6, 0], [6, 2, 3], [0, 3, 2933]]).astype(dtype)
    pi = np.array([0.70532947, 0.00109989, 0.29357064])
    P_mle = tmatrix(C, reversible=True, mu=pi)

    N = 1000
    sampler = TransitionMatrixSampler(C, reversible=True, mu=pi, n_steps=10)

    sample = np.zeros((N, 3, 3))
    for i in range(N):
        s = sampler.sample()
        np.testing.assert_equal(s.dtype, dtype)
        sample[i, :, :] = s
    mean = np.mean(sample, axis=0)
    std = np.std(sample, axis=0)

    # Check if sample mean and MLE agree within the sample standard deviation
    np.testing.assert_(np.all(np.abs(mean - P_mle) <= std))
