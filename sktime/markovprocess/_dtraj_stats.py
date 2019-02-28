
import numpy as np

from msmtools import estimation as msmest
from pyemma.util.linalg import submatrix

__author__ = 'noe'


# TODO: this could me moved to msmtools.dtraj
def blocksplit_dtrajs(dtrajs, lag=1, sliding=True, shift=None):
    """ Splits the discrete trajectories into approximately uncorrelated fragments

    Will split trajectories into fragments of lengths lag or longer. These fragments
    are overlapping in order to conserve the transition counts at given lag.
    If sliding=True, the resulting trajectories will lead to exactly the same count
    matrix as when counted from dtrajs. If sliding=False (sampling at lag), the
    count matrices are only equal when also setting shift=0.

    Parameters
    ----------
    dtrajs : list of ndarray(int)
        Discrete trajectories
    lag : int
        Lag time at which counting will be done. If sh
    sliding : bool
        True for splitting trajectories for sliding count, False if lag-sampling will be applied
    shift : None or int
        Start of first full tau-window. If None, shift will be randomly generated

    """
    dtrajs_new = []
    for dtraj in dtrajs:
        if len(dtraj) <= lag:
            continue
        if shift is None:
            s = np.random.randint(min(lag, dtraj.size - lag))
        else:
            s = shift
        if sliding:
            if s > 0:
                dtrajs_new.append(dtraj[0:lag + s])
            for t0 in range(s, dtraj.size - lag, lag):
                dtrajs_new.append(dtraj[t0:t0 + 2 * lag])
        else:
            for t0 in range(s, dtraj.size - lag, lag):
                dtrajs_new.append(dtraj[t0:t0 + lag + 1])
    return dtrajs_new


# TODO: this could me moved to msmtools.dtraj
def cvsplit_dtrajs(dtrajs):
    """ Splits the trajectories into a training and test set with approximately equal number of trajectories

    Parameters
    ----------
    dtrajs : list of ndarray(int)
        Discrete trajectories

    """
    if len(dtrajs) == 1:
        raise ValueError('Only have a single trajectory. Cannot be split into train and test set')
    I0 = np.random.choice(len(dtrajs), int(len(dtrajs) / 2), replace=False)
    I1 = np.array(list(set(list(np.arange(len(dtrajs)))) - set(list(I0))))
    dtrajs_train = [dtrajs[i] for i in I0]
    dtrajs_test = [dtrajs[i] for i in I1]
    return dtrajs_train, dtrajs_test


class DiscreteTrajectoryStats(object):
    r""" Statistics, count matrices and connectivity from discrete trajectories

    Operates sparse by default.

    """

    def __init__(self, dtrajs):
        # TODO: extensive input checking!
        from pyemma.util.types import ensure_dtraj_list

        # discrete trajectories
        self._dtrajs = ensure_dtraj_list(dtrajs)

        ## basic count statistics
        # histogram
        from msmtools.dtraj import count_states
        self._hist = count_states(self._dtrajs, ignore_negative=True)
        # total counts
        self._total_count = np.sum(self._hist)
        # number of states
        self._nstates = msmest.number_of_states(dtrajs)

        # not yet estimated
        self._counted_at_lag = False

    def to_coreset(self, core_set, in_place=True):
        """

        Parameters
        ----------
        core_set: an array of micro-states to include as core-sets

        in_place: boolean, default=True
            if True, replace the current dtrajs
            if False, return a copy

        Returns
        -------
        dtrajs
        """
        import copy
        dtrajs = self._dtrajs if in_place else copy.deepcopy(self._dtrajs)

        core_set = np.array(core_set, dtype=int)
        # build a boolean expression to create a mask of indices within the core set.
        expr = ['(d == {i})'.format(i=i) for i in core_set]
        expr = '|'.join(expr)

        def to_ranges(a):
            # return a list of consecutive ranges in array a.
            cons = np.split(a, np.where(np.diff(a) != 1)[0] + 1)
            ranges = [(np.min(x), np.max(x) + 1) if len(x) > 1
                      else (x[0], x[0] + 1) for x in cons]
            return ranges

        for d in dtrajs:
            within_core_set = eval(expr)
            outside_core_set = np.logical_not(within_core_set)
            inds_outside_set = np.where(outside_core_set)[0]
            # determine ranges to update, which lies outside the core set.
            ranges = to_ranges(inds_outside_set)

            # start with first valid core set value.
            for start, stop in ranges:
                core_set = d[start - 1] if start > 0 else -1
                d[start:stop] = core_set

        # re-initialize
        if in_place:
            self.__init__(dtrajs)

        return dtrajs

    @staticmethod
    def _compute_connected_sets(C, mincount_connectivity, strong=True):
        """ Computes the connected sets of C.

        C : count matrix
        mincount_connectivity : float
            Minimum count which counts as a connection.
        strong : boolean
            True: Seek strongly connected sets. False: Seek weakly connected sets.
        Returns
        -------
        Cconn, S
        """
        import msmtools.estimation as msmest
        import scipy.sparse as scs
        if scs.issparse(C):
            Cconn = C.tocsr(copy=True)
            Cconn.data[Cconn.data < mincount_connectivity] = 0
            Cconn.eliminate_zeros()
        else:
            Cconn = C.copy()
            Cconn[np.where(Cconn < mincount_connectivity)] = 0

        # treat each connected set separately
        S = msmest.connected_sets(Cconn, directed=strong)
        return S

    def count_lagged(self, lag, count_mode='sliding', mincount_connectivity='1/n', show_progress=True, n_jobs=None,
                     name=''):
        r""" Counts transitions at given lag time

        Parameters
        ----------
        lag : int
            lagtime in trajectory steps

        count_mode : str, optional, default='sliding'
            mode to obtain count matrices from discrete trajectories. Should be one of:

            * 'sliding' : A trajectory of length T will have :math:`T-\tau` counts
              at time indexes
              .. math:: (0 \rightarray \tau), (1 \rightarray \tau+1), ..., (T-\tau-1 \rightarray T-1)

            * 'effective' : Uses an estimate of the transition counts that are
              statistically uncorrelated. Recommended when used with a
              Bayesian MSM.

            * 'sample' : A trajectory of length T will have :math:`T / \tau` counts
              at time indexes
              .. math:: (0 \rightarray \tau), (\tau \rightarray 2 \tau), ..., (((T/tau)-1) \tau \rightarray T)

        show_progress: bool, default=True
            show the progress for the expensive effective count mode computation.

        n_jobs: int or None

        """
        # store lag time
        self._lag = lag

        # Compute count matrix
        count_mode = count_mode.lower()
        if count_mode == 'sliding':
            self._C = msmest.count_matrix(self._dtrajs, lag, sliding=True)
        elif count_mode == 'sample':
            self._C = msmest.count_matrix(self._dtrajs, lag, sliding=False)
        elif count_mode == 'effective':
            from pyemma.util.reflection import getargspec_no_self
            argspec = getargspec_no_self(msmest.effective_count_matrix)
            kw = {}
            from pyemma.util.contexts import nullcontext
            ctx = nullcontext()
            if 'callback' in argspec.args:  # msmtools effective cmatrix ready for multiprocessing?
                from pyemma._base.progress import ProgressReporter
                from pyemma._base.parallel import get_n_jobs

                kw['n_jobs'] = get_n_jobs() if n_jobs is None else n_jobs

                if show_progress:
                    pg = ProgressReporter()
                    # this is a fast operation
                    C_temp = msmest.count_matrix(self._dtrajs, lag, sliding=True)
                    pg.register(C_temp.nnz, '{}: compute stat. inefficiencies'.format(name), stage=0)
                    del C_temp
                    kw['callback'] = pg.update
                    ctx = pg.context(stage=0)
            with ctx:
                self._C = msmest.effective_count_matrix(self._dtrajs, lag, **kw)
        else:
            raise ValueError('Count mode ' + count_mode + ' is unknown.')

        # store mincount_connectivity
        if mincount_connectivity == '1/n':
            mincount_connectivity = 1.0 / np.shape(self._C)[0]
        self._mincount_connectivity = mincount_connectivity

        # Compute reversibly connected sets
        if self._mincount_connectivity > 0:
            self._connected_sets = \
                self._compute_connected_sets(self._C, mincount_connectivity=self._mincount_connectivity)
        else:
            self._connected_sets = msmest.connected_sets(self._C)

        # set sizes and count matrices on reversibly connected sets
        self._connected_set_sizes = np.zeros((len(self._connected_sets)))
        self._C_sub = np.empty((len(self._connected_sets)), dtype=np.object)
        for i in range(len(self._connected_sets)):
            # set size
            self._connected_set_sizes[i] = len(self._connected_sets[i])
            # submatrix
            # self._C_sub[i] = submatrix(self._C, self._connected_sets[i])

        # largest connected set
        self._lcs = self._connected_sets[0]

        # if lcs has no counts, make lcs empty
        if submatrix(self._C, self._lcs).sum() == 0:
            self._lcs = np.array([], dtype=int)

        # mapping from full to lcs
        self._full2lcs = -1 * np.ones((self._nstates), dtype=int)
        self._full2lcs[self._lcs] = np.arange(len(self._lcs))

        # remember that this function was called
        self._counted_at_lag = True

    # ==================================
    # Permanent properties
    # ==================================

    def _assert_counted_at_lag(self):
        """
        Checks if count_lagged has been run
        """
        assert self._counted_at_lag, \
            "You haven't run count_lagged yet. Do that first before accessing lag-based quantities"

    def _assert_subset(self, A):
        """
        Checks if set A is a subset of states

        Parameters
        ----------
        A : int or int array
            set of states
        """
        if np.size(A) == 0:
            return True  # empty set is always contained
        assert np.max(A) < self._nstates, 'Chosen set contains states that are not included in the data.'

    @property
    def nstates(self):
        """
        Number (int) of states
        """
        return self._nstates

    @property
    def discrete_trajectories(self):
        """
        A list of integer arrays with the original (unmapped) discrete trajectories:

        """
        return self._dtrajs

    @property
    def total_count(self):
        """
        Total number of counts

        """
        return self._hist.sum()

    @property
    def histogram(self):
        r""" Histogram of discrete state counts

        """
        return self._hist

    # ==================================
    # Estimated properties
    # ==================================

    @property
    def lagtime(self):
        """
        The active set of states on which all computations and estimations will be done

        """
        self._assert_counted_at_lag()
        return self._lag

    def count_matrix(self, connected_set=None, subset=None, effective=False):
        """The count matrix

        Parameters
        ----------
        connected_set : int or None, optional, default=None
            connected set index. See :func:`connected_sets` to get a sorted list of connected sets.
            This parameter is exclusive with subset.
        subset : array-like of int or None, optional, default=None
            subset of states to compute the count matrix on. This parameter is exclusive with subset.
        effective : bool, optional, default=False
            Statistically uncorrelated transition counts within the active set of states.

            You can use this count matrix for any kind of estimation, in particular it is meant to give reasonable
            error bars in uncertainty measurements (error perturbation or Gibbs sampling of the posterior).

            The effective count matrix is obtained by dividing the sliding-window count matrix by the lag time. This
            can be shown to provide a likelihood that is the geometrical average over shifted subsamples of the trajectory,
            :math:`(s_1,\:s_{tau+1},\:...),\:(s_2,\:t_{tau+2},\:...),` etc. This geometrical average converges to the
            correct likelihood in the statistical limit [1]_.

        References
        ----------

        ..[1] Trendelkamp-Schroer B, H Wu, F Paul and F Noe. 2015:
            Reversible Markov models of molecular kinetics: Estimation and uncertainty.
            in preparation.
        """
        self._assert_counted_at_lag()
        if subset is not None and connected_set is not None:
            raise ValueError('Can\'t set both connected_set and subset.')
        if subset is not None:
            self._assert_subset(subset)
            C = submatrix(self._C, subset)
        elif connected_set is not None:
            C = submatrix(self._C, self._connected_sets[connected_set])
        else:  # full matrix wanted
            C = self._C

        # effective count matrix wanted?
        if effective:
            C /= float(self._lag)

        return C

    def histogram_lagged(self, connected_set=None, subset=None, effective=False):
        r""" Histogram of discrete state counts

        """
        C = self.count_matrix(connected_set=connected_set, subset=subset, effective=effective)
        return C.sum(axis=1)

    @property
    def total_count_lagged(self, connected_set=None, subset=None, effective=False):
        h = self.histogram_lagged(connected_set=connected_set, subset=subset, effective=effective)
        return h.sum()

    @property
    def count_matrix_largest(self, effective=False):
        """The count matrix on the largest connected set

        """
        return self.count_matrix(connected_set=0, effective=effective)

    @property
    def largest_connected_set(self):
        """
        The largest reversible connected set of states

        """
        self._assert_counted_at_lag()
        return self._lcs

    @property
    def visited_set(self):
        r""" The set of visited states
        """
        from sktime.markovprocess.util import visited_set
        return visited_set(self._dtrajs)

    @property
    def connected_sets(self):
        """
        The reversible connected sets of states, sorted by size (descending)

        """
        self._assert_counted_at_lag()
        return self._connected_sets

    @property
    def connected_set_sizes(self):
        """The numbers of states for each connected set

        """
        self._assert_counted_at_lag()
        return self._connected_set_sizes
