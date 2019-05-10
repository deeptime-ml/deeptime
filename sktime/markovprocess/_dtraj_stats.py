import numpy as np

from msmtools import estimation as msmest
from msmtools.dtraj import count_states

from pyemma.util.linalg import submatrix

from sktime.markovprocess import Q_
from sktime.markovprocess.util import visited_set

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


class TransitionCountingMixin(object):
    r""" Statistics, count matrices and connectivity from discrete trajectories

    Operates sparse by default.

    """

    @property
    def lagtime(self) -> Q_:
        """ The lag time at which the Markov model was estimated."""
        return self._lag

    @lagtime.setter
    def lagtime(self, value: [int, str]):
        self._lag = Q_(value)

    @property
    def nstates_full(self):
        """ Number of states in discrete trajectories """
        return self._nstates_full

    @property
    def active_set(self):
        """The active set of states on which all computations and estimations will be done"""
        return self._active_set

    @active_set.setter
    def active_set(self, value):
        self._active_set = value

    @property
    def dt_traj(self) -> Q_:
        """Time interval between discrete steps of the time series."""
        return self.timestep_traj

    @dt_traj.setter
    def dt_traj(self, value: [int, str]):
        self.timestep_traj = Q_(value)

    @property
    def largest_connected_set(self):
        """The largest reversible connected set of states."""
        return self._connected_sets[0] if self._connected_sets is not None else ()

    @property
    def connected_sets(self):
        """The reversible connected sets of states, sorted by size (descending)."""
        return self._connected_sets

    # TODO: ever used?
    def map_discrete_trajectories_to_active(self, dtrajs):
        """
        A list of integer arrays with the discrete trajectories mapped to the connectivity mode used.
        For example, for connectivity='largest', the indexes will be given within the connected set.
        Frames that are not in the connected set will be -1.
        """
        # compute connected dtrajs
        dtrajs_active = []
        # TODO: checking for same set?
        for dtraj in dtrajs:
            dtrajs_active.append(self._full2lcs[dtraj])

        return dtrajs_active

    @property
    def count_matrix_active(self):
        """The count matrix on the active set given the connectivity mode used.

        For example, for connectivity='largest', the count matrix is given only on the largest reversibly connected set.

        Attention: This count matrix has been obtained by sliding a window of length tau across the data. It contains
        a factor of tau more counts than are statistically uncorrelated. It's fine to use this matrix for maximum
        likelihood estimated, but it will give far too small errors if you use it for uncertainty calculations. In order
        to do uncertainty calculations, use the effective count matrix, see:
        :attr:`effective_count_matrix`

        See Also
        --------
        effective_count_matrix
            For a count matrix with effective (statistically uncorrelated) counts.

        """
        return self._C_active

    @property
    def count_matrix_full(self):
        """
        The count matrix on full set of discrete states, irrespective as to whether they are connected or not.
        Attention: This count matrix has been obtained by sliding a window of length tau across the data. It contains
        a factor of tau more counts than are statistically uncorrelated. It's fine to use this matrix for maximum
        likelihood estimated, but it will give far too small errors if you use it for uncertainty calculations. In order
        to do uncertainty calculations, use the effective count matrix, see: :attr:`effective_count_matrix`
        (only implemented on the active set), or divide this count matrix by tau.

        See Also
        --------
        effective_count_matrix
            For a active-set count matrix with effective (statistically uncorrelated) counts.

        """
        # TODO: used to be _C_full
        return self._C

    @property
    def active_state_fraction(self):
        """The fraction of states in the largest connected set."""
        return float(self._nstates) / float(self._nstates_full)

    @property
    def active_count_fraction(self):
        """The fraction of counts in the largest connected set."""
        from pyemma.util.discrete_trajectories import count_states

        hist = count_states(self._hist)
        hist_active = hist[self.active_set]
        return float(np.sum(hist_active)) / float(np.sum(hist))

    @property
    # TODO: needed to be computed in fit()
    def active_state_indexes(self):
        """Ensures that the connected states are indexed and returns the indices."""
        if not hasattr(self, '_active_state_indexes'):
            from pyemma.util.discrete_trajectories import index_states
            self._active_state_indexes = index_states(self.discrete_trajectories_active)
        return self._active_state_indexes

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

    def _compute_count_matrix(self, dtrajs, count_mode, mincount_connectivity='1/n'):
        r""" Counts transitions at given lag time

        Parameters
        ----------

        dtrajs

        """
        from pyemma.util.types import ensure_dtraj_list

        # discrete trajectories
        dtrajs = ensure_dtraj_list(dtrajs)

        ## basic count statistics
        # histogram
        self._hist = count_states(dtrajs, ignore_negative=True)
        # total counts
        self._total_count = np.sum(self._hist)
        # number of states
        self._nstates = msmest.number_of_states(dtrajs)
        self._visited_set = visited_set(dtrajs)

        # TODO: compute this in discrete steps of traj
        lag = int((self.lagtime / self.dt_traj).magnitude)

        # Compute count matrix
        if count_mode == 'sliding':
            self._C = msmest.count_matrix(dtrajs, lag, sliding=True)
        elif count_mode == 'sample':
            self._C = msmest.count_matrix(dtrajs, lag, sliding=False)
        elif count_mode == 'effective':
            self._C = msmest.effective_count_matrix(dtrajs, lag)
        else:
            raise ValueError('Count mode {} is unknown.'.format(count_mode))
        self._nstates_full = np.shape(self._C)[0]

        # store mincount_connectivity
        if mincount_connectivity == '1/n':
            mincount_connectivity = 1.0 / np.shape(self._C)[0]

        # Compute reversibly connected sets
        if mincount_connectivity > 0:
            self._connected_sets = \
                self._compute_connected_sets(self._C, mincount_connectivity=mincount_connectivity)
        else:
            self._connected_sets = msmest.connected_sets(self._C)

        # set sizes of reversibly connected sets
        self._connected_set_sizes = np.zeros(len(self._connected_sets))
        for i in range(len(self._connected_sets)):
            # set size
            self._connected_set_sizes[i] = len(self._connected_sets[i])

        # largest connected set
        self._lcs = self._connected_sets[0]

        # if lcs has no counts, make lcs empty
        if submatrix(self._C, self._lcs).sum() == 0:
            self._lcs = np.empty(0, dtype=int)

        # mapping from full to lcs
        self._full2lcs = -1 * np.ones(self._nstates, dtype=int)
        self._full2lcs[self._lcs] = np.arange(len(self._lcs))

    @property
    def nstates(self) -> int:
        """Number of states """
        return self._nstates

    @property
    def total_count(self):
        """Total number of counts"""
        return self._hist.sum()

    @property
    def histogram(self):
        """ Histogram of discrete state counts"""
        return self._hist

    def count_matrix(self, connected_set=None, subset=None, effective=False):
        r"""The count matrix

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
        if subset is not None and connected_set is not None:
            raise ValueError('Can\'t set both connected_set and subset.')
        if subset is not None:
            if np.size(subset) > 0:
                assert np.max(subset) < self._nstates, 'Chosen set contains states that are not included in the data.'
            C = submatrix(self._C, subset)
        elif connected_set is not None:
            C = submatrix(self._C, self._connected_sets[connected_set])
        else:  # full matrix wanted
            C = self._C

        # effective count matrix wanted?
        # FIXME: this modifies self._C in place!
        if effective:
            C /= float(self._lag)

        return C

    def histogram_lagged(self, connected_set=None, subset=None, effective=False):
        r""" Histogram of discrete state counts"""
        C = self.count_matrix(connected_set=connected_set, subset=subset, effective=effective)
        return C.sum(axis=1)

    @property
    def total_count_lagged(self, connected_set=None, subset=None, effective=False):
        h = self.histogram_lagged(connected_set=connected_set, subset=subset, effective=effective)
        return h.sum()

    @property
    def visited_set(self):
        """ The set of visited states"""
        return self._visited_set

    @property
    def connected_set_sizes(self):
        """The numbers of states for each connected set"""
        return self._connected_set_sizes
