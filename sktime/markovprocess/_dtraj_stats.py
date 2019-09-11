import numpy as np
from msmtools import estimation as msmest
from msmtools.dtraj import count_states
from sklearn.utils.random import check_random_state

from sktime.base import Estimator, Model
from sktime.markovprocess import Q_
from sktime.util import submatrix

__author__ = 'noe'


# TODO: this could me moved to msmtools.dtraj
def blocksplit_dtrajs(dtrajs, lag=1, sliding=True, shift=None, random_state=None):
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
    random_state = check_random_state(random_state)
    for dtraj in dtrajs:
        if len(dtraj) <= lag:
            continue
        if shift is None:
            s = random_state.randint(min(lag, dtraj.size - lag))
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
def cvsplit_dtrajs(dtrajs, random_state=None):
    """ Splits the trajectories into a training and test set with approximately equal number of trajectories

    Parameters
    ----------
    dtrajs : list of ndarray(int)
        Discrete trajectories

    """
    if len(dtrajs) == 1:
        raise ValueError('Only have a single trajectory. Cannot be split into train and test set')
    random_state = check_random_state(random_state)
    I0 = random_state.choice(len(dtrajs), int(len(dtrajs) / 2), replace=False)
    I1 = np.array(list(set(list(np.arange(len(dtrajs)))) - set(list(I0))))
    dtrajs_train = [dtrajs[i] for i in I0]
    dtrajs_test = [dtrajs[i] for i in I1]
    return dtrajs_train, dtrajs_test


class TransitionCountModel(Model):
    r""" Statistics, count matrices and connectivity from discrete trajectories

    Operates sparse by default.

    """

    def __init__(self, lagtime=1, active_set=None, dt_traj='1 step',
                 connected_sets=(), count_matrix=None, hist=None) -> None:
        self._lag = Q_(lagtime)
        self._active_set = active_set
        self._dt_traj = Q_(dt_traj) if isinstance(dt_traj, (str, int)) else dt_traj
        self._connected_sets = connected_sets
        self._C = count_matrix
        self._hist = hist

        if count_matrix is not None:
            self._nstates_full = count_matrix.shape[0]
        else:
            self._nstates_full = 0

        # mapping from full to lcs
        if active_set is not None:
            self._full2lcs = -1 * np.ones(self.nstates, dtype=int)
            self._full2lcs[active_set] = np.arange(len(active_set))
        else:
            self._full2lcs = None

    @property
    def lagtime(self) -> Q_:
        """ The lag time at which the Markov model was estimated."""
        return self._lag

    @property
    def active_set(self):
        """The active set of states on which all computations and estimations will be done"""
        return self._active_set

    @property
    def dt_traj(self) -> Q_:
        """Time interval between discrete steps of the time series."""
        return self._dt_traj

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
        from sktime.markovprocess.sample import ensure_dtraj_list
        dtrajs = ensure_dtraj_list(dtrajs)
        dtrajs_active = [
            self._full2lcs[dtraj]
            for dtraj in dtrajs
        ]
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
        return self.subselect_count_matrix(subset=self.active_set)

    # todo rename to count_matrix
    @property
    def count_matrix(self):
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
        return self._C

    @property
    def active_state_fraction(self):
        """The fraction of states in the largest connected set."""
        return float(self.nstates) / float(self._nstates_full)

    @property
    def active_count_fraction(self):
        """The fraction of counts in the largest connected set."""
        hist_active = self._hist[self.active_set]
        return float(np.sum(hist_active)) / float(np.sum(self._hist))

    @property
    def nstates(self) -> int:
        """Number of states """
        return self.count_matrix.shape[0]

    @property
    def nstates_active(self) -> int:
        """Number of states in the active set"""
        return len(self._active_set)

    @property
    def total_count(self):
        """Total number of counts"""
        return self._hist.sum()

    @property
    def state_histogram(self):
        """ Histogram of discrete state counts"""
        return self._hist

    # todo: rename to subselect_count_matrix
    def subselect_count_matrix(self, connected_set=None, subset=None, effective=False):
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
            J. Chem. Phys. 143, 174101 (2015); https://doi.org/10.1063/1.4934536
        """
        if subset is not None and connected_set is not None:
            raise ValueError('Can\'t set both connected_set and subset.')
        if subset is not None:
            if np.size(subset) > 0:
                assert np.max(subset) < self.nstates, 'Chosen set contains states that are not included in the data.'
            C = submatrix(self._C, subset)
        elif connected_set is not None:
            C = submatrix(self._C, self._connected_sets[connected_set])
        else:  # full matrix wanted
            C = self._C

        # effective count matrix wanted?
        if effective:
            C = C.copy()
            C /= float(self._lag)

        return C

    def histogram_lagged(self, connected_set=None, subset=None, effective=False):
        r""" Histogram of discrete state counts"""
        C = self.subselect_count_matrix(connected_set=connected_set, subset=subset, effective=effective)
        return C.sum(axis=1)

    @property
    def total_count_lagged(self, connected_set=None, subset=None, effective=False):
        h = self.histogram_lagged(connected_set=connected_set, subset=subset, effective=effective)
        return h.sum()

    @property
    def visited_set(self):
        """ The set of visited states"""
        return np.argwhere(self._hist > 0)[:, 0]

    @property
    def connected_set_sizes(self):
        # set sizes of reversibly connected sets
        return np.array([len(x) for x in self.connected_sets])

    @property
    def effective_count_matrix(self):
        return self.subselect_count_matrix(connected_set=self.active_set, effective=True)


class TransitionCountEstimator(Estimator):

    def __init__(self,lagtime: int = 1, count_mode: str = 'sliding', mincount_connectivity='1/n', dt_traj='1',
                 stationary_dist_constraint=None):
        self.lagtime = lagtime
        self.count_mode = count_mode
        self.mincount_connectivity = mincount_connectivity
        self.dt_traj = dt_traj
        self.stationary_dist_constraint = stationary_dist_constraint
        super().__init__()

    @property
    def dt_traj(self):
        return self._dt_traj

    @dt_traj.setter
    def dt_traj(self, value):
        self._dt_traj = Q_(value)

    def _create_model(self) -> TransitionCountModel:
        return TransitionCountModel()

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

    # TODO: this has been moved from MSM estimaton here, because it fits better, but is still too MSM-est like (input of pi, instead of active_set)
    @staticmethod
    def _prepare_input_revpi(C, pi):
        """Max. state index visited by trajectories"""
        nC = C.shape[0]
        # Max. state index of the stationary vector array
        npi = pi.shape[0]
        # pi has to be defined on all states visited by the trajectories
        if nC > npi:
            raise ValueError('There are visited states for which no stationary probability is given')
        # Reduce pi to the visited set
        pi_visited = pi[0:nC]
        # Find visited states with positive stationary probabilities"""
        pos = np.where(pi_visited > 0.0)[0]
        # Reduce C to positive probability states"""
        C_pos = msmest.connected_cmatrix(C, lcc=pos)
        # Compute largest connected set of C_pos, undirected connectivity"""
        lcc = msmest.largest_connected_set(C_pos, directed=False)
        return pos[lcc]

    def fit(self, data):
        r""" Counts transitions at given lag time

        Parameters
        ----------

        dtrajs : list of 1D numpy arrays containing integers

        """
        if not isinstance(data, (list, tuple)):
            data = [data]

        # typecheck
        for x in data:
            assert isinstance(x, np.ndarray), "dtraj list contained element which was not a numpy array"
            assert x.ndim == 1, "dtraj list contained multi-dimensional array"
            assert issubclass(x.dtype.type, np.integer), "dtraj list contained non-integer array"

        assert isinstance(self._model, TransitionCountModel)

        # these are now discrete trajectories
        dtrajs = data

        ## basic count statistics
        # histogram
        hist = count_states(dtrajs, ignore_negative=True)

        # Compute count matrix
        count_mode = self.count_mode
        lagtime = self.lagtime
        if count_mode == 'sliding':
            count_matrix = msmest.count_matrix(dtrajs, lagtime, sliding=True)
        elif count_mode == 'sample':
            count_matrix = msmest.count_matrix(dtrajs, lagtime, sliding=False)
        elif count_mode == 'effective':
            count_matrix = msmest.effective_count_matrix(dtrajs, lagtime)
        else:
            raise ValueError('Count mode {} is unknown.'.format(count_mode))

        # store mincount_connectivity
        if self.mincount_connectivity == '1/n':
            self.mincount_connectivity = 1.0 / np.shape(count_matrix)[0]

        # Compute reversibly connected sets
        if self.mincount_connectivity > 0:
            connected_sets = self._compute_connected_sets(count_matrix,
                                                          mincount_connectivity=self.mincount_connectivity)
        else:
            connected_sets = msmest.connected_sets(count_matrix)

        if self.stationary_dist_constraint is not None:
            active_set = self._prepare_input_revpi(count_matrix, self.stationary_dist_constraint)
        else:
            # largest connected set
            active_set = connected_sets[0]

        # if active set has no counts, make it empty
        if submatrix(count_matrix, active_set).sum() == 0:
            active_set = np.empty(0, dtype=int)

        self._model.__init__(
            lagtime=lagtime, active_set=active_set, dt_traj=self.dt_traj,
            connected_sets=connected_sets, count_matrix=count_matrix,
            hist=hist
        )

        return self
