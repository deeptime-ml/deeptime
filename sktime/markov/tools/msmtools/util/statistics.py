
# This file is part of MSMTools.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# MSMTools is free software: you can redistribute it and/or modify
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

'''
Created on Jul 25, 2014

@author: noe
'''


import numpy as np
import math
import itertools
import warnings

from . import types


def _confidence_interval_1d(data, alpha):
    """ Computes the mean and alpha-confidence interval of the given sample set

    Parameters
    ----------
    data : ndarray
        a 1D-array of samples
    alpha : float in [0,1]
        the confidence level, i.e. percentage of data included in the interval

    Returns
    -------
    (m, l, r) : m is the mean of the data, and (l, r) are the m-alpha/2
        and m+alpha/2 confidence interval boundaries.
    """
    # CHECK INPUT
    if alpha < 0 or alpha > 1:
        raise ValueError('Not a meaningful confidence level: '+str(alpha))
    # exception: if data are constant, return three times the constant and raise a warning
    dmin = np.min(data)
    dmax = np.max(data)
    if dmin == dmax:
        warnings.warn('confidence interval for constant data is not meaningful')
        return dmin, dmin, dmin

    # COMPUTE INTERVAL
    # compute mean
    m = np.mean(data)
    # sort data
    sdata = np.sort(data)
    # index of the mean
    im = np.searchsorted(sdata, m)
    if im == 0 or im == len(sdata):
        pm = im
    else:
        pm = (im-1) + (m-sdata[im-1])/(sdata[im]-sdata[im-1])
    # left interval boundary
    pl = pm - alpha*pm
    il1 = max(0, int(math.floor(pl)))
    il2 = min(len(sdata)-1, int(math.ceil(pl)))
    l = sdata[il1] + (pl - il1)*(sdata[il2] - sdata[il1])
    # right interval boundary
    pr = pm + alpha*(len(data)-im)
    ir1 = max(0, int(math.floor(pr)))
    ir2 = min(len(sdata)-1, int(math.ceil(pr)))
    r = sdata[ir1] + (pr - ir1)*(sdata[ir2] - sdata[ir1])

    # return
    return m, l, r


def _indexes(arr):
    """ Returns the list of all indexes of the given array.

    Currently works for one and two-dimensional arrays

    """
    myarr = np.array(arr)
    if myarr.ndim == 1:
        return range(len(myarr))
    elif myarr.ndim == 2:
        return itertools.product(range(arr.shape[0]), range(arr.shape[1]))
    else:
        raise NotImplementedError('Only supporting arrays of dimension 1 and 2 as yet.')


def _column(arr, indexes):
    """ Returns a column with given indexes from a deep array

    For example, if the array is a matrix and indexes is a single int, will
    return arr[:,indexes]. If the array is an order 3 tensor and indexes is a
    pair of ints, will return arr[:,indexes[0],indexes[1]], etc.

    """
    if arr.ndim == 2 and types.is_int(indexes):
        return arr[:, indexes]
    elif arr.ndim == 3 and len(indexes) == 2:
        return arr[:, indexes[0], indexes[1]]
    else:
        raise NotImplementedError('Only supporting arrays of dimension 2 and 3 as yet.')


def confidence_interval(data, conf=0.95):
    r""" Computes element-wise confidence intervals from a sample of ndarrays

    Given a sample of arbitrarily shaped ndarrays, computes element-wise
    confidence intervals

    Parameters
    ----------
    data : array-like of dimension 1 to 3
        array of numbers or arrays. The first index is used as the sample
        index, the remaining indexes are specific to the array of interest
    conf : float, optional, default = 0.95
        confidence interval

    Return
    ------
    lower : ndarray(shape)
        element-wise lower bounds
    upper : ndarray(shape)
        element-wise upper bounds

    """
    if conf < 0 or conf > 1:
        raise ValueError('Not a meaningful confidence level: '+str(conf))

    try:
        data = types.ensure_ndarray(data, kind='numeric')
    except:
        # if 1D array of arrays try to fuse it
        if isinstance(data, np.ndarray) and np.ndim(data) == 1:
            newshape = tuple([len(data)] + list(data[0].shape))
            newdata = np.zeros(newshape)
            for i in range(len(data)):
                newdata[i, :] = data[i]
            data = newdata

    types.assert_array(data, kind='numeric')

    if np.ndim(data) == 1:
        m, lower, upper = _confidence_interval_1d(data, conf)
        return lower, upper
    else:
        I = _indexes(data[0])
        lower = np.zeros(data[0].shape)
        upper = np.zeros(data[0].shape)
        for i in I:
            col = _column(data, i)
            m, lower[i], upper[i] = _confidence_interval_1d(col, conf)
        # return
        return lower, upper


def _maxlength(X):
    """ Returns the maximum length of signal trajectories X """
    return np.fromiter((map(lambda x: len(x), X)), dtype=int).max()


def statistical_inefficiency(X, truncate_acf=True, mact=1.0):
    r""" Estimates the statistical inefficiency from univariate time series X

    The statistical inefficiency [1]_ is a measure of the correlatedness of samples in a signal.
    Given a signal :math:`{x_t}` with :math:`N` samples and statistical inefficiency :math:`I \in (0,1]`, there are
    only :math:`I \cdot N` effective or uncorrelated samples in the signal. This means that :math:`I \cdot N` should
    be used in order to compute statistical uncertainties. See [2]_ for a review.

    The statistical inefficiency is computed as :math:`I = (2 \tau)^{-1}` using the damped autocorrelation time

    .. math:
        \tau = \frac{1}{2}+\sum_{K=1}^{N} A(k) \left(1-\frac{k}{N}\right)

    where

    .. math:
        A(k) = \frac{\langle x_t x_{t+k} \rangle_t - \langle x^2 \rangle_t}{\mathrm{var}(x)}

    is the autocorrelation function of the signal :math:`{x_t}`, which is computed either for a single or multiple
    trajectories.

    Parameters
    ----------
    X : float array or list of float arrays
        Univariate time series (single or multiple trajectories)
    truncate_acf : bool, optional, default=True
        When the normalized autocorrelation function passes through 0, it is truncated in order to avoid integrating
        random noise

    References
    ----------
    .. [1] Anderson, T. W.: The Statistical Analysis of Time Series (Wiley, New York, 1971)

    .. [2] Janke, W: Statistical Analysis of Simulations: Data Correlations and Error Estimation
        Quantum Simulations of Complex Many-Body Systems: From Theory to Algorithms, Lecture Notes,
        J. Grotendorst, D. Marx, A. Muramatsu (Eds.), John von Neumann Institute for Computing, Juelich
        NIC Series 10, pp. 423-445, 2002.

    """
    # check input
    assert np.ndim(X[0]) == 1, 'Data must be 1-dimensional'
    N = _maxlength(X)  # max length
    # mean-free data
    xflat = np.concatenate(X)
    Xmean = np.mean(xflat)
    X0 = [x-Xmean for x in X]
    # moments
    x2m = np.mean(xflat ** 2)
    del xflat, Xmean
    # integrate damped autocorrelation
    corrsum = 0.0
    for lag in range(N):
        acf = 0.0
        n = 0
        # cache partial sums
        for x in X0:
            Nx = len(x)  # length of this trajectory
            if Nx > lag:  # only use trajectories that are long enough
                prod = x[:Nx-lag] * x[lag:]
                acf += np.sum(prod)
                n += Nx-lag
        acf /= float(n)
        if acf <= 0 and truncate_acf:  # zero autocorrelation. Exit
            break
        elif lag > 0:  # start integrating at lag 1 (effect of lag 0 is contained in the 0.5 below
            corrsum += acf * (1.0 - (float(lag)/float(N)))
    # compute damped correlation time
    corrtime = 0.5 + mact * corrsum / x2m
    # return statistical inefficiency
    return 1.0 / (2 * corrtime)
