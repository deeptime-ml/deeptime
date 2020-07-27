
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

r"""This module implements effective transition counts

.. moduleauthor:: F. Noe <frank DOT noe AT fu-berlin DOT de>

"""

import numpy as np
import scipy.sparse

from ...util.statistics import statistical_inefficiency
from .count_matrix import count_matrix_coo2_mult
from ...dtraj import number_of_states

from scipy.sparse.csr import csr_matrix

__author__ = 'noe, marscher'


def _split_sequences_singletraj(dtraj, nstates, lag):
    """ splits the discrete trajectory into conditional sequences by starting state

    Parameters
    ----------
    dtraj : int-iterable
        discrete trajectory
    nstates : int
        total number of discrete states
    lag : int
        lag time

    """
    sall = [[] for _ in range(nstates)]
    res_states = []
    res_seqs = []

    for t in range(len(dtraj)-lag):
        sall[dtraj[t]].append(dtraj[t+lag])

    for i in range(nstates):
        if len(sall[i]) > 0:
            res_states.append(i)
            res_seqs.append(np.array(sall[i]))
    return res_states, res_seqs


def _split_sequences_multitraj(dtrajs, lag):
    """ splits the discrete trajectories into conditional sequences by starting state

    Parameters
    ----------
    dtrajs : list of int-iterables
        discrete trajectories
    nstates : int
        total number of discrete states
    lag : int
        lag time

    """
    n = number_of_states(dtrajs)
    res = []
    for i in range(n):
        res.append([])
    for dtraj in dtrajs:
        states, seqs = _split_sequences_singletraj(dtraj, n, lag)
        for i in range(len(states)):
            res[states[i]].append(seqs[i])
    return res


def _indicator_multitraj(ss, i, j):
    """ Returns conditional sequence for transition i -> j given all conditional sequences """
    iseqs = ss[i]
    res = []
    for iseq in iseqs:
        x = np.zeros(len(iseq))
        I = np.where(iseq == j)
        x[I] = 1.0
        res.append(x)
    return res


def _wrapper(*args):
    # writes results of statistical_inefficiency to destination memmap (array_fn)
    seqs, truncate_acf, mact, I, J, array_fn, start, stop = args[0]
    array = np.memmap(array_fn, mode='r+', dtype=np.float64)
    partial = np.empty(len(I))
    for n, (i, j) in enumerate(zip(I, J)):
         s = _indicator_multitraj(seqs, i, j)
         partial[n] = statistical_inefficiency(s, truncate_acf=truncate_acf, mact=mact)
    array[start:stop] = partial


class _arguments_generator(object):
    # splits I and J in blocks to serve n_jobs processes.
    def __init__(self, I, J, splitted_seqs, truncate_acf, mact, array, njobs):
        self.I = I
        self.J = J
        self.splitted_seqs = splitted_seqs
        self.truncate_acf = truncate_acf
        self.mact = mact
        self.array = array
        self.njobs = njobs

    def __len__(self):
        return len(self.I)

    def n_blocks(self):
        k = 4
        n = max(len(self.I) // (k*self.njobs - 1), 1)
        return n

    def __iter__(self):
        def chunks(n):
            for i in range(0, len(self.I), n):
                yield self.I[i:i + n], self.J[i:i+n]

        start = 0
        for n, (I, J) in enumerate(chunks(self.n_blocks())):
            stop = start + len(I)
            yield (self.splitted_seqs, self.truncate_acf, self.mact, I, J, self.array, start, stop)
            start = stop


def statistical_inefficiencies(dtrajs, lag, C=None, truncate_acf=True, mact=2.0, n_jobs=1, callback=None):
    r""" Computes statistical inefficiencies of sliding-window transition counts at given lag

    Consider a discrete trajectory :math`{ x_t }` with :math:`x_t \in {1, ..., n}`. For each starting state :math:`i`,
    we collect the target sequence

    .. mathh:
        Y^(i) = {x_{t+\tau} | x_{t}=i}

    which contains the time-ordered target states at times :math:`t+\tau` whenever we started in state :math:`i`
    at time :math:`t`. Then we define the indicator sequence:

    .. math:
        a^{(i,j)}_t (\tau) = 1(Y^(i)_t = j)

    The statistical inefficiency for transition counts :math:`c_{ij}(tau)` is computed as the statistical inefficiency
    of the sequence :math:`a^{(i,j)}_t (\tau)`.

    Parameters
    ----------
    dtrajs : list of int-iterables
        discrete trajectories
    lag : int
        lag time
    C : scipy sparse matrix (n, n) or None
        sliding window count matrix, if already available
    truncate_acf : bool, optional, default=True
        When the normalized autocorrelation function passes through 0, it is truncated in order to avoid integrating
        random noise
    n_jobs: int, default=1
        If greater one, the function will be evaluated with multiple processes.
    callback: callable, default=None
        will be called for every statistical inefficiency computed (number of nonzero elements in count matrix).
        If n_jobs is greater one, the callback will be invoked per finished batch.

    Returns
    -------
    I : scipy sparse matrix (n, n)
        Statistical inefficiency matrix with a sparsity pattern identical to the sliding-window count matrix at the
        same lag time. Will contain a statistical inefficiency :math:`I_{ij} \in (0,1]` whenever there is a count
        :math:`c_{ij} > 0`. When there is no transition count (:math:`c_{ij} = 0`), the statistical inefficiency is 0.

    See also
    --------
    msmtools.util.statistics.statistical_inefficiency
        used to compute the statistical inefficiency for conditional trajectories

    """
    # count matrix
    if C is None:
        C = count_matrix_coo2_mult(dtrajs, lag, sliding=True, sparse=True)
    if callback is not None:
        if not callable(callback):
            raise ValueError('Provided callback is not callable')
    # split sequences
    splitseq = _split_sequences_multitraj(dtrajs, lag)
    # compute inefficiencies
    I, J = C.nonzero()
    if n_jobs > 1:
        from multiprocessing.pool import Pool, MapResult
        from contextlib import closing
        import tempfile

        # to avoid pickling partial results, we store these in a numpy.memmap
        ntf = tempfile.NamedTemporaryFile(delete=False)
        arr = np.memmap(ntf.name, dtype=np.float64, mode='w+', shape=C.nnz)
        #arr[:] = np.nan
        gen = _arguments_generator(I, J, splitseq, truncate_acf=truncate_acf, mact=truncate_acf,
                                   array=ntf.name, njobs=n_jobs)
        if callback:
            x = gen.n_blocks()
            _callback = lambda _: callback(x)
        else:
            _callback = callback
        with closing(Pool(n_jobs)) as pool:
            result_async = [pool.apply_async(_wrapper, (args,), callback=_callback)
                            for args in gen]

            [t.get() for t in result_async]
            data = np.array(arr[:])
            #assert np.all(np.isfinite(data))
        import os
        os.unlink(ntf.name)
    else:
        data = np.empty(C.nnz)
        for index, (i, j) in enumerate(zip(I, J)):
            data[index] = statistical_inefficiency(_indicator_multitraj(splitseq, i, j),
                                                   truncate_acf=truncate_acf, mact=mact)
            if callback is not None:
                callback(1)
    res = csr_matrix((data, (I, J)), shape=C.shape)
    return res


def effective_count_matrix(dtrajs, lag, average='row', truncate_acf=True, mact=1.0, n_jobs=1, callback=None):
    r""" Computes the statistically effective transition count matrix

    Given a list of discrete trajectories, compute the effective number of statistically uncorrelated transition
    counts at the given lag time. First computes the full sliding-window counts :math:`c_{ij}(tau)`. Then uses
    :func:`statistical_inefficiencies` to compute statistical inefficiencies :math:`I_{ij}(tau)`. The number of
    effective counts in a row is then computed as

    .. math:
        c_i^{\mathrm{eff}}(tau) = \sum_j I_{ij}(tau) c_{ij}(tau)

    and the effective transition counts are obtained by scaling the rows accordingly:

    .. math:
        c_{ij}^{\mathrm{eff}}(tau) = \frac{c_i^{\mathrm{eff}}(tau)}{c_i(tau)} c_{ij}(tau)

    This procedure is not yet published, but a manuscript is in preparation [1]_.

    Parameters
    ----------
    dtrajs : list of int-ndarrays
        discrete trajectories
    lag : int
        lag time
    average : str, default='row'
        Use either of 'row', 'all', 'none', with the following consequences:
        'none': the statistical inefficiency is applied separately to each
            transition count (not recommended)
        'row': the statistical inefficiency is averaged (weighted) by row
            (recommended).
        'all': the statistical inefficiency is averaged (weighted) over all
            transition counts (not recommended).
    truncate_acf : bool, optional, default=True
        Mode of estimating the autocorrelation time of transition counts.
        True: When the normalized autocorrelation function passes through 0,
        it is truncated in order to avoid integrating random noise. This
        tends to lead to a slight underestimate of the autocorrelation time
    mact : float, default=2.0
        multiplier for the autocorrelation time. We tend to underestimate the
        autocorrelation time (and thus overestimate effective counts)
        because the autocorrelation function is truncated when it passes
        through 0 in order to avoid numerical instabilities.
        This is a purely heuristic factor trying to compensate this effect.
        This parameter might be removed in the future when a more robust
        estimation method of the autocorrelation time is used.

    n_jobs: int, default=1
        If greater one, the function will be evaluated with multiple processes.

    callback: callable, default=None
        will be called for every statistical inefficiency computed (number of nonzero elements in count matrix).
        If n_jobs is greater one, the callback will be invoked per finished batch.

    See also
    --------
    statistical_inefficiencies
        is used for computing the statistical inefficiences of sliding window
        transition counts

    References
    ----------
    .. [1] Noe, F. and H. Wu: in preparation (2015)

    """
    # observed C
    C = count_matrix_coo2_mult(dtrajs, lag, sliding=True, sparse=True)
    # statistical inefficiencies
    si = statistical_inefficiencies(dtrajs, lag, C=C, truncate_acf=truncate_acf, mact=mact,
                                    n_jobs=n_jobs, callback=callback)
    # effective element-wise counts
    Ceff = C.multiply(si)
    # averaging
    if average.lower() == 'row':
        # reduction factor by row
        factor = np.array(Ceff.sum(axis=1) / np.maximum(1.0, C.sum(axis=1)))
        # row-based effective counts
        Ceff = scipy.sparse.csr_matrix(C.multiply(factor))
    elif average.lower() == 'all':
        # reduction factor by all
        factor = Ceff.sum() / C.sum()
        Ceff = scipy.sparse.csr_matrix(C.multiply(factor))
    # else: by element, we're done.

    return Ceff
