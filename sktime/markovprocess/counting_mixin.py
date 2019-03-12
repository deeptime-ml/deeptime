import numpy as np

from sktime.markovprocess import Q_


class MSMCountingMixin(object):

    def __new__(cls, *args, **kwargs):
        self = super(MSMCountingMixin, cls).__new__(*args, **kwargs)
        self._nstates_full = float('inf')
        self._full2active = None
        self._C_active = None
        self._C_full = None
        self._nstates_full = None
        self._nstates_active = None
        self.active_set = None
        self._nstates_active = None
        self._nstates_full = None

        self._connected_sets = ()
        return self

    @property
    def lagtime(self) -> Q_:
        """ The lag time at which the Markov model was estimated."""
        return self._lag

    @lagtime.setter
    def lagtime(self, value: [int, str]):
        self._lag = Q_(value)

    @property
    def nstates_full(self):
        r""" Number of states in discrete trajectories """
        return self._nstates_full

    @property
    def active_set(self):
        """
        The active set of states on which all computations and estimations will be done

        """
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
            dtrajs_active.append(self._full2active[dtraj])

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
        return self._C_full

    @property
    def active_state_fraction(self):
        """The fraction of states in the largest connected set."""
        return float(self._nstates) / float(self._nstates_full)

    @property
    def active_count_fraction(self):
        """The fraction of counts in the largest connected set."""
        from pyemma.util.discrete_trajectories import count_states

        hist = count_states(self._dtrajs_full)
        hist_active = hist[self.active_set]
        return float(np.sum(hist_active)) / float(np.sum(hist))

    # TODO: change to statistically effective count matrix!
    def effective_count_matrix(self, dtrajs):
        """Statistically uncorrelated transition counts within the active set of states

        You can use this count matrix for Bayesian estimation or error perturbation.

        References
        ----------
        [1] Noe, F. (2015) Statistical inefficiency of Markov model count matrices
            http://publications.mi.fu-berlin.de/1699/1/autocorrelation_counts.pdf

        """
        from msmtools.estimation import effective_count_matrix
        Ceff_full = effective_count_matrix(dtrajs, self.lagtime)
        from pyemma.util.linalg import submatrix
        Ceff = submatrix(Ceff_full, self.active_set)
        return Ceff
        # return self._C_active / float(self.lagtime)

    @property
    def active_state_indexes(self):
        """Ensures that the connected states are indexed and returns the indices."""
        if not hasattr(self, '_active_state_indexes'):
            from pyemma.util.discrete_trajectories import index_states
            self._active_state_indexes = index_states(self.discrete_trajectories_active)
        return self._active_state_indexes
