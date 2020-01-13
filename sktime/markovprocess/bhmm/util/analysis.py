# This file is part of BHMM (Bayesian Hidden Markov Models).
#
# Copyright (c) 2016 Frank Noe (Freie Universitaet Berlin)
# and John D. Chodera (Memorial Sloan-Kettering Cancer Center, New York)
#
# BHMM is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np


def beta_confidence_intervals(ci_X, ntrials, ci=0.95):
    """
    Compute confidence intervals of beta distributions.

    Parameters
    ----------
    ci_X : numpy.array
        Computed confidence interval estimate from `ntrials` experiments
    ntrials : int
        The number of trials that were run.
    ci : float, optional, default=0.95
        Confidence interval to report (e.g. 0.95 for 95% confidence interval)

    Returns
    -------
    Plow : float
        The lower bound of the symmetric confidence interval.
    Phigh : float
        The upper bound of the symmetric confidence interval.

    Examples
    --------

    >>> ci_X = np.random.rand(10,10)
    >>> ntrials = 100
    >>> Plow, Phigh = beta_confidence_intervals(ci_X, ntrials)

    """
    # Compute low and high confidence interval for symmetric CI about mean.
    ci_low = 0.5 - ci / 2
    ci_high = 0.5 + ci / 2

    # Compute for every element of ci_X.
    from scipy.stats import beta
    Plow = ci_X * 0.0
    Phigh = ci_X * 0.0
    for i in range(ci_X.shape[0]):
        for j in range(ci_X.shape[1]):
            Plow[i, j] = beta.ppf(ci_low, a=ci_X[i, j] * ntrials, b=(1 - ci_X[i, j]) * ntrials)
            Phigh[i, j] = beta.ppf(ci_high, a=ci_X[i, j] * ntrials, b=(1 - ci_X[i, j]) * ntrials)

    return Plow, Phigh


def empirical_confidence_interval(sample, interval=0.95):
    """
    Compute specified symmetric confidence interval for empirical sample.

    Parameters
    ----------
    sample : numpy.array
        The empirical samples.
    interval : float, optional, default=0.95
        Size of desired symmetric confidence interval (0 < interval < 1)
        e.g. 0.68 for 68% confidence interval, 0.95 for 95% confidence interval

    Returns
    -------
    low : float
        The lower bound of the symmetric confidence interval.
    high : float
        The upper bound of the symmetric confidence interval.

    Examples
    --------
    >>> sample = np.random.randn(1000)
    >>> low, high = empirical_confidence_interval(sample)

    >>> low, high = empirical_confidence_interval(sample, interval=0.65)

    >>> low, high = empirical_confidence_interval(sample, interval=0.99)

    """
    # Sort sample in increasing order.
    sample = np.sort(sample)

    # Determine sample size.
    N = len(sample)

    # Compute low and high indices.
    low_index = int(np.round((N - 1) * (0.5 - interval / 2))) + 1
    high_index = int(np.round((N - 1) * (0.5 + interval / 2))) + 1

    # Compute low and high.
    low = sample[low_index]
    high = sample[high_index]

    return low, high
