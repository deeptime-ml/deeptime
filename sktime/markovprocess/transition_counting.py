from typing import Union, Optional, List

import numpy as np
from msmtools import estimation as msmest
from scipy.sparse import coo_matrix

from sktime.base import Estimator, Model
from sktime.markovprocess import Q_
from sktime.markovprocess.util import count_states
from sktime.util import submatrix, ensure_dtraj_list

__author__ = 'noe, clonker'


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
    from sklearn.utils.random import check_random_state
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
    from sklearn.utils.random import check_random_state
    if len(dtrajs) == 1:
        raise ValueError('Only have a single trajectory. Cannot be split into train and test set')
    random_state = check_random_state(random_state)
    I0 = random_state.choice(len(dtrajs), int(len(dtrajs) / 2), replace=False)
    I1 = np.array(list(set(list(np.arange(len(dtrajs)))) - set(list(I0))))
    dtrajs_train = [dtrajs[i] for i in I0]
    dtrajs_test = [dtrajs[i] for i in I1]
    return dtrajs_train, dtrajs_test


class TransitionCountModel(Model):
    r""" Statistics, count matrices, and connectivity from discrete trajectories.
    """

    def __init__(self, count_matrix: Union[np.ndarray, coo_matrix], counting_mode: str, lagtime: int,
                 state_histogram: Optional[np.ndarray], dt_traj: Union[str, int] = '1 step',
                 state_symbols: Optional[np.ndarray] = None,
                 count_matrix_full: Union[None, np.ndarray, coo_matrix] = None,
                 state_histogram_full: Optional[np.ndarray] = None):
        r"""Creates a new TransitionCountModel. This can be used to, e.g., construct Markov state models.

        Parameters
        ----------
        count_matrix : array_like
            The count matrix. In case it was estimated with 'sliding', it contains a factor of `lagtime` more counts
            than are statistically uncorrelated.
        counting_mode : str
            One of 'sliding', 'sample', or 'effective'. Indicates the counting method that was used to estimate the
            count matrix. In case of 'sliding', a sliding window of the size of the lagtime was used to
            count transitions. It therefore contains a factor of `lagtime` more counts than are statistically
            uncorrelated. It's fine to use this matrix for maximum likelihood estimation, but it will give far too
            small errors if you use it for uncertainty calculations. In order to do uncertainty calculations,
            use the effective count matrix, see: :attr:`effective_count_matrix`, divide this count matrix by tau, or
            use 'effective' as estimation parameter.
        lagtime : int
            The time offset which was used to count transitions in state.
        state_histogram : array_like
            Histogram over the visited states in discretized trajectories.
        dt_traj : str or int, default='1 step'
            time step
        state_symbols : array_like, optional, default=None
            Symbols of the original discrete trajectory that are represented in the counting model. If None, the
            symbols are assumed to represent the data, i.e., a iota range over the number of states. Subselection
            of the model also subselects the symbols.
        count_matrix_full : array_like, optional, default=None
            Count matrix for all state symbols. If None, the count matrix provided as first argument is assumed to
            take that role.
        state_histogram_full : array_like, optional, default=None
            Histogram over all state symbols. If None, the provided state_histogram  is assumed to take that role.
        """

        if count_matrix is None or not isinstance(count_matrix, (np.ndarray, coo_matrix)):
            raise ValueError("count matrix needs to be an ndarray but was {}".format(count_matrix))

        self._count_matrix = count_matrix
        self._counting_mode = counting_mode
        self._lag = Q_(lagtime)
        self._dt_traj = Q_(dt_traj) if isinstance(dt_traj, (str, int)) else dt_traj
        self._state_histogram = state_histogram

        if state_symbols is None:
            # if symbols is not set, assume that the count matrix represents all states in the data
            state_symbols = np.arange(self.n_states)

        if len(state_symbols) != self.n_states:
            raise ValueError("Number of symbols in counting model must coincide with the number of states in the "
                             "count matrix! (#symbols = {}, #states = {})".format(len(state_symbols), self.n_states))
        self._state_symbols = state_symbols
        if count_matrix_full is None:
            count_matrix_full = count_matrix
        self._count_matrix_full = count_matrix_full
        if self.n_states_full < self.n_states:
            # full number of states must be at least as large as n_states
            raise ValueError("Number of states was bigger than full number of "
                             "states. (#states = {}, #states_full = {}), likely a wrong "
                             "full count matrix.".format(self.n_states, self.n_states_full))
        if state_histogram_full is None:
            state_histogram_full = state_histogram
        if self.n_states_full != len(state_histogram_full):
            raise ValueError("Mismatch between number of states represented in full state histogram and full "
                             "count matrix (#states histogram = {}, #states matrix = {})"\
                .format(len(state_histogram_full), self.n_states_full))
        self._state_histogram_full = state_histogram_full

    @property
    def state_histogram_full(self):
        r""" Histogram over all states in the trajectories. """
        return self._state_histogram_full

    @property
    def n_states_full(self) -> int:
        r""" Full number of states represented in the underlying data. """
        return self.count_matrix_full.shape[0]

    @property
    def state_symbols(self) -> np.ndarray:
        r""" Symbols (states) that are represented in this count model. """
        return self._state_symbols

    @property
    def counting_mode(self) -> str:
        """ The counting mode that was used to estimate the contained count matrix.
        One of 'sliding', 'sample', 'effective'.
        """
        return self._counting_mode

    @property
    def lagtime(self) -> Q_:
        """ The lag time at which the Markov model was estimated."""
        return self._lag

    @property
    def dt_traj(self) -> Q_:
        """Time interval between discrete steps of the time series."""
        return self._dt_traj

    @property
    def is_full_model(self) -> bool:
        r""" Can be used to determine whether this counting model refers to the full model that represents all states
        of the data.

        Returns
        -------
        whether this counting model represents all states of the data
        """
        return self.n_states == self.n_states_full

    def transform_discrete_trajectories_to_symbols(self, dtrajs):
        r"""A list of integer arrays with the discrete trajectories mapped to the currently used set of symbols.
        For example, if there has been a subselection of the model for connectivity='largest', the indices will be
        given within the connected set, frames that do not correspond to a considered symbol are set to -1.

        Parameters
        ----------
        dtrajs : array_like or list of array_like
            discretized trajectories

        Returns
        -------
        Curated discretized trajectories so that unconsidered symbols are mapped to -1.
        """

        if self.is_full_model:
            # no-op
            return dtrajs
        else:
            dtrajs = ensure_dtraj_list(dtrajs)
            mapping = -1 * np.ones(self.n_states_full, dtype=np.int32)
            mapping[self.state_symbols] = np.arange(self.n_states)
            return [mapping[dtraj] for dtraj in dtrajs]

    @property
    def count_matrix(self):
        """The count matrix, possibly restricted to a subset of states.

        Attention: This count matrix could have been obtained by sliding a window of length tau across the data.
        It then contains a factor of tau more counts than are statistically uncorrelated. It's fine to use this matrix
        for maximum likelihood estimation, but it will give far too small errors if you use it for uncertainty
        calculations. In order to do uncertainty calculations, use the effective count matrix,
        see: :attr:`effective_count_matrix` (only implemented on the active set), or divide this count matrix by tau.

        See Also
        --------
        effective_count_matrix
            For a active-set count matrix with effective (statistically uncorrelated) counts.

        """
        return self._count_matrix

    @property
    def count_matrix_full(self):
        r""" The count matrix on full set of discrete states, irrespective as to whether they are selected or not.
        """
        return self._count_matrix_full

    @property
    def active_state_fraction(self):
        """The fraction of states represented in this count model."""
        return float(self.n_states) / float(self.n_states_full)

    @property
    def active_count_fraction(self):
        """The fraction of counts represented in this count model."""
        return float(np.sum(self.state_histogram)) / float(np.sum(self.state_histogram_full))

    @property
    def n_states(self) -> int:
        """Number of states """
        return self.count_matrix.shape[0]

    @property
    def total_count(self):
        """Total number of counts"""
        return self._state_histogram.sum()

    @property
    def state_histogram(self):
        """ Histogram of discrete state counts"""
        return self._state_histogram

    def connected_sets(self, mincount_connectivity: Union[None, float] = None) -> List[np.ndarray]:
        r""" Computes the connected sets of the counting matrix. A threshold can be set fixing a number of counts
        required to consider two states connected. In case of sliding window the number of counts is increased by a
        factor of `lagtime`. In case of 'effective' counting, the number of sliding window counts were divided by
        the lagtime

        Parameters
        ----------
        mincount_connectivity : float, optional, default=None
            Number of counts required to consider two states connected. In case of sliding/sample counting mode,
            the default corresponds to 0, in case of effective counting mode the default corresponds to 1/n_states,
            where n_states refers to the full amount of states present in the data.
        Returns
        -------
        A list of arrays containing integers (states), each array representing a connected set. The list is
        ordered decreasingly by the size of the individual components.
        """
        from sktime.markovprocess.bhmm.estimators import _tmatrix_disconnected
        if mincount_connectivity is None:
            if self.counting_mode == 'sliding' or self.counting_mode == 'sample':
                mincount_connectivity = 0.
            elif self.counting_mode == 'effective':
                mincount_connectivity = 1. / float(self.n_states_full)
            else:
                raise RuntimeError("Counting mode was not one of 'sliding', 'sample', "
                                   "'effective': {}".format(self.counting_mode))
        return _tmatrix_disconnected.connected_sets(self.count_matrix,
                                                    mincount_connectivity=mincount_connectivity,
                                                    strong=True)

    def submodel(self, states: np.ndarray):
        r"""This returns a count model that is restricted to a selection of states.

        Parameters
        ----------
        states : array_like
            The states to restrict to.

        Returns
        -------

        """
        if np.max(states) >= self.n_states:
            raise ValueError("Tried restricting model to states that are not represented! "
                             "States range from 0 to {}.".format(np.max(states)))
        sub_count_matrix = submatrix(self.count_matrix, states)
        sub_symbols = self.state_symbols[states]
        sub_state_histogram = self.state_histogram[states]
        return TransitionCountModel(sub_count_matrix, self.counting_mode, self.lagtime, sub_state_histogram,
                                    state_symbols=sub_symbols, dt_traj=self.dt_traj,
                                    count_matrix_full=self.count_matrix_full,
                                    state_histogram_full=self.state_histogram_full)

    def _subselect_count_matrix(self, connected_set=None, subset=None, effective=False):
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
                assert np.max(subset) < self.n_states, 'Chosen set contains states that are not included in the data.'
            C = submatrix(self._count_matrix, subset)
        elif connected_set is not None:
            C = submatrix(self._count_matrix, self._connected_sets[connected_set])
        else:  # full matrix wanted
            C = self._count_matrix

        # effective count matrix wanted?
        if effective:
            C = C.copy()
            C /= float(self._lag)

        return C

    def histogram_lagged(self, connected_set=None, subset=None, effective=False):
        r""" Histogram of discrete state counts"""
        C = self._subselect_count_matrix(connected_set=connected_set, subset=subset, effective=effective)
        return C.sum(axis=1)

    @property
    def total_count_lagged(self, connected_set=None, subset=None, effective=False):
        h = self.histogram_lagged(connected_set=connected_set, subset=subset, effective=effective)
        return h.sum()

    @property
    def visited_set(self):
        """ The set of visited states. """
        return np.argwhere(self.state_histogram > 0)[:, 0]

    @property
    def connected_set_sizes(self):
        # set sizes of reversibly connected sets
        return np.array([len(x) for x in self.connected_sets])

    @property
    def effective_count_matrix(self):
        return self._subselect_count_matrix(effective=True)

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
        if mincount_connectivity > 0:
            if scs.issparse(C):
                Cconn = C.tocsr(copy=True)
                Cconn.data[Cconn.data < mincount_connectivity] = 0
                Cconn.eliminate_zeros()
            else:
                Cconn = C.copy()
                Cconn[np.where(Cconn < mincount_connectivity)] = 0
        else:
            Cconn = C
        # treat each connected set separately
        S = msmest.connected_sets(Cconn, directed=strong)
        return S

    @staticmethod
    def states_revpi(C, pi):
        r"""
        Compute states so that the subselected model is defined on the intersection of the states with positive
        stationary vector and the largest connected set (undirected).

        Parameters
        ----------
        C : (M, M) ndarray
            count matrix
        pi : (M,) ndarray
            stationary vector on full set of states

        Returns
        -------
        active set
        """
        nC = C.shape[0]
        # Max. state index of the stationary vector array
        npi = pi.shape[0]
        # pi has to be defined on all states visited by the trajectories
        if nC > npi:
            raise ValueError('There are visited states for which no stationary probability is given')
        # Reduce pi to the visited set
        pi_visited = pi[:nC]
        # Find visited states with positive stationary probabilities"""
        pos = np.where(pi_visited > 0.0)[0]
        # Reduce C to positive probability states"""
        C_pos = msmest.largest_connected_submatrix(C, lcc=pos)
        # Compute largest connected set of C_pos, undirected connectivity"""
        lcc = msmest.largest_connected_set(C_pos, directed=False)
        return pos[lcc]


class TransitionCountEstimator(Estimator):

    def __init__(self, lagtime: int, count_mode: str = 'sliding', dt_traj='1',
                 stationary_dist_constraint=None):
        super().__init__()
        self.lagtime = lagtime
        self.count_mode = count_mode
        self.dt_traj = dt_traj
        self.stationary_dist_constraint = stationary_dist_constraint

    @property
    def dt_traj(self):
        return self._dt_traj

    @dt_traj.setter
    def dt_traj(self, value):
        self._dt_traj = Q_(value)

    def fetch_model(self) -> TransitionCountModel:
        return self._model

    def fit(self, data, **kw):
        r""" Counts transitions at given lag time

        Parameters
        ----------

        dtrajs : array_like or list of array_like
            discretized trajectories

        """
        dtrajs = ensure_dtraj_list(data)

        # basic count statistics
        histogram = count_states(dtrajs, ignore_negative=True)

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

        self._model = TransitionCountModel(
            lagtime=lagtime, dt_traj=self.dt_traj, count_matrix=count_matrix, state_histogram=histogram
        )

        return self
