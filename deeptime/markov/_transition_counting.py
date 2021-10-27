from functools import wraps
from typing import Union, Optional, List, Callable

import numpy as np
import scipy
from scipy.sparse import coo_matrix, issparse

from .tools import estimation as msmest
from .tools.analysis import is_connected
from ..base import Model, EstimatorTransformer
from ..util.matrix import submatrix
from ..util.types import ensure_dtraj_list

__author__ = 'noe, clonker'


def requires_state_histogram(func: Callable) -> Callable:
    """
    Decorator marking properties which can only be evaluated if the TransitionCountModel contains statistics over the
    data, i.e., a state histogram.

    Parameters
    ----------
    func : Callable
        the to-be decorated property

    Returns
    -------
    wrapper : Callable
        the decorated property
    """
    @wraps(func)
    def wrap(self, *args, **kw):
        if self.state_histogram is None:
            raise RuntimeError("The model was not provided with a state histogram, this property cannot be evaluated.")
        return func(self, *args, **kw)

    return wrap


class TransitionCountModel(Model):
    r""" Statistics, count matrices, and connectivity from discrete trajectories. These statistics can be used to, e.g.,
    construct MSMs. This model can create submodels (see :func:`submodel`)
    that are restricted to a certain selection of states. This subselection can be made by

    * analyzing the connected sets of the
      count matrix (:func:`connected_sets`)
    * pruning states by thresholding with a mincount_connectivity parameter,
    * or simply providing a subset of states manually.

    Parameters
    ----------
    count_matrix : (N, N) ndarray or sparse matrix
        The count matrix. In case it was estimated with 'sliding', it contains a factor of `lagtime` more counts
        than are statistically uncorrelated.
    counting_mode : str, optional, default=None
        If not None, one of 'sliding', 'sample', or 'effective'.
        Indicates the counting method that was used to estimate the count matrix. In case of 'sliding', a sliding
        window of the size of the lagtime was used to count transitions. It therefore contains a factor
        of `lagtime` more counts than are statistically uncorrelated. It's fine to use this matrix for maximum
        likelihood estimation, but it will give far too small errors if you use it for uncertainty calculations.
        In order to do uncertainty calculations, use the effective count matrix, see
        :attr:`effective_count_matrix`, divide this count matrix by tau, or use 'effective' as estimation parameter.
    lagtime : int, optional, default=1
        The time offset which was used to count transitions in state.
    state_histogram : array_like, optional, default=None
        Histogram over the visited states in discretized trajectories.
    state_symbols : array_like, optional, default=None
        Symbols of the original discrete trajectory that are represented in the counting model. If None, the
        symbols are assumed to represent the data, i.e., a iota range over the number of states. Subselection
        of the model also subselects the symbols.
    count_matrix_full : array_like, optional, default=None
        Count matrix for all state symbols. If None, the count matrix provided as first argument is assumed to
        take that role.
    state_histogram_full : array_like, optional, default=None
        Histogram over all state symbols. If None, the provided state_histogram  is assumed to take that role.

    See Also
    --------
    TransitionCountEstimator
    """

    def __init__(self, count_matrix, counting_mode: Optional[str] = None, lagtime: int = 1,
                 state_histogram: Optional[np.ndarray] = None,
                 state_symbols: Optional[np.ndarray] = None, count_matrix_full=None,
                 state_histogram_full: Optional[np.ndarray] = None):
        super().__init__()
        if count_matrix is None:
            raise ValueError("count matrix was None")
        if not issparse(count_matrix):
            count_matrix = np.asarray(count_matrix)
        if (not issparse(count_matrix) and np.any(count_matrix < 0)) or count_matrix.min() < 0:
            raise ValueError("Count matrix elements must all be non-negative")

        self._count_matrix = count_matrix
        self._counting_mode = counting_mode
        self._lag = lagtime
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
        if state_histogram_full is not None and self.n_states_full != len(state_histogram_full):
            raise ValueError(
                "Mismatch between number of states represented in full state histogram and full count matrix "
                "(#states histogram = {}, #states matrix = {})".format(len(state_histogram_full), self.n_states_full)
            )
        self._state_histogram_full = state_histogram_full

    @property
    def state_histogram_full(self) -> Optional[np.ndarray]:
        r""" Histogram over all states in the trajectories. """
        return self._state_histogram_full

    @property
    def n_states_full(self) -> int:
        r""" Full number of states represented in the underlying data. """
        return self.count_matrix_full.shape[0]

    @property
    def states(self) -> np.ndarray:
        r""" The states in this model, i.e., a iota range from 0 (inclusive) to :meth:`n_states` (exclusive).
        See also: :meth:`state_symbols`. """
        return np.arange(self.n_states)

    @property
    def state_symbols(self) -> np.ndarray:
        r""" Symbols (states) that are represented in this count model. """
        return self._state_symbols

    @property
    def counting_mode(self) -> Optional[str]:
        """ The counting mode that was used to estimate the contained count matrix.
        One of 'None', 'sliding', 'sample', 'effective'.
        """
        return self._counting_mode

    @property
    def lagtime(self) -> int:
        """ The lag time at which the Markov model was estimated."""
        return self._lag

    @property
    def is_full_model(self) -> bool:
        r""" Determine whether this counting model refers to the full model that represents all states of the data.

        Returns
        -------
        is_full_model : bool
            Whether this counting model represents all states of the data.
        """
        return self.n_states == self.n_states_full

    def transform_discrete_trajectories_to_submodel(self, dtrajs):
        r"""A list of integer arrays with the discrete trajectories mapped to the currently used set of symbols.
        For example, if there has been a subselection of the model for connectivity='largest', the indices will be
        given within the connected set, frames that do not correspond to a considered symbol are set to -1.

        Parameters
        ----------
        dtrajs : array_like or list of array_like
            discretized trajectories

        Returns
        -------
        array_like or list of array_like
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
        calculations. In order to do uncertainty calculations, use effective counting during estimation,
        or divide this count matrix by tau.
        """
        return self._count_matrix

    @property
    def count_matrix_full(self):
        r""" The count matrix on full set of discrete states, irrespective as to whether they are selected or not.
        """
        return self._count_matrix_full

    @property
    def selected_state_fraction(self) -> float:
        """The fraction of states represented in this count model."""
        return float(self.n_states) / float(self.n_states_full)

    @property
    @requires_state_histogram
    def selected_count_fraction(self) -> float:
        """The fraction of counts represented in this count model."""
        return float(np.sum(self.state_histogram)) / float(np.sum(self.state_histogram_full))

    @property
    def n_states(self) -> int:
        """Number of states """
        return self.count_matrix.shape[0]

    @property
    @requires_state_histogram
    def total_count(self) -> int:
        """Total number of counts"""
        return self._state_histogram.sum()

    @property
    @requires_state_histogram
    def visited_set(self) -> np.ndarray:
        """ The set of visited states. """
        return np.argwhere(self.state_histogram > 0)[:, 0]

    @property
    def state_histogram(self) -> Optional[np.ndarray]:
        """ Histogram of discrete state counts, can be None in case no statistics were provided """
        return self._state_histogram

    def connected_sets(self, connectivity_threshold: float = 0., directed: bool = True,
                       probability_constraint: Optional[np.ndarray] = None,
                       sort_by_population: bool = False) -> List[np.ndarray]:
        r""" Computes the connected sets of the counting matrix. A threshold can be set fixing a number of counts
        required to consider two states connected. In case of sliding window the number of counts is increased by a
        factor of `lagtime`. In case of 'sliding-effective' counting, the number of sliding window counts were
        divided by the lagtime and can therefore also be in the open interval (0, 1). Same for 'effective' counting.

        Parameters
        ----------
        connectivity_threshold : float, optional, default=0.
            Number of counts required to consider two states connected. When the count matrix was estimated with
            effective mode or sliding-effective mode, a threshold of :math:`1 / \mathrm{n_states_full}` is
            commonly used.
        directed : bool, optional, default=True
            Compute connected set for directed or undirected transition graph, default directed
        probability_constraint : (N,) ndarray, optional, default=None
            constraint on the whole state space, sets all counts to zero which have no probability
        sort_by_population : bool, optional, default=False
            This flag can be used to order the resulting list of sets in decreasing order by the most counts.

        Returns
        -------
        A list of arrays containing integers (states), each array representing a connected set. The list is
        ordered decreasingly by the size of the individual components.
        """
        from deeptime.markov import compute_connected_sets
        count_matrix = self.count_matrix
        if probability_constraint is not None:
            # pi has to be defined on all states visited by the trajectories
            if len(probability_constraint) != self.n_states_full:
                raise ValueError("The connected sets with a constraint can only be evaluated if the constraint "
                                 "refers to the whole state space (#states total = {}), but it had a length of "
                                 "#constrained states = {}".format(self.n_states_full, len(probability_constraint)))
            probability_constraint = probability_constraint[self.state_symbols]

            # Find visited states with positive stationary probabilities
            pos = np.where(probability_constraint <= 0.0)[0]
            count_matrix = count_matrix.copy()

            if scipy.sparse.issparse(count_matrix):
                count_matrix = count_matrix.tolil()
            count_matrix[pos, :] = 0.
            count_matrix[:, pos] = 0.
        connected_sets = compute_connected_sets(count_matrix, connectivity_threshold, directed=directed)
        if sort_by_population:
            score = np.array([self.count_matrix[np.ix_(s, s)].sum() for s in connected_sets])
            # want decreasing order therefore sort by -1 * score
            connected_sets = [connected_sets[i] for i in np.argsort(-score)]
        return connected_sets

    def states_to_symbols(self, states: np.ndarray) -> np.ndarray:
        r""" Converts a list of states to a list of symbols which can be related back to the original data.

        Parameters
        ----------
        states : (N,) ndarray
            Array of states.

        Returns
        -------
        symbols : (N,) ndarray
            Array of symbols.
        """
        return self.state_symbols[states]

    def symbols_to_states(self, symbols):
        r"""
        Converts a set of symbols to state indices in this count model instance. The symbols which
        are no longer present in this model are discarded. It can happen that the order is
        changed or the result is smaller than the input length.

        Parameters
        ----------
        symbols : array_like
            the symbols to be mapped to state indices

        Returns
        -------
        states : ndarray
            An array of states.
        """
        # only take symbols which are still present in this model
        if isinstance(symbols, set):
            symbols = list(symbols)
        symbols = np.intersect1d(np.asarray(symbols), self.state_symbols)
        return np.argwhere(np.isin(self.state_symbols, symbols)).flatten()

    def submodel(self, states: np.ndarray):
        r"""This returns a count model that is restricted to a selection of states.

        Parameters
        ----------
        states : array_like
            The states to restrict to.

        Returns
        -------
        submodel : TransitionCountModel
            A submodel restricted to the requested states.
        """
        states = np.atleast_1d(states)
        if np.max(states) >= self.n_states:
            raise ValueError("Tried restricting model to states that are not represented! "
                             "States range from 0 to {}.".format(np.max(states)))
        sub_count_matrix = submatrix(self.count_matrix, states)
        if self.state_symbols is not None:
            sub_symbols = self.state_symbols[states]
        else:
            sub_symbols = None
        if self.state_histogram is not None:
            sub_state_histogram = self.state_histogram[states]
        else:
            sub_state_histogram = None
        return TransitionCountModel(sub_count_matrix, self.counting_mode, self.lagtime, sub_state_histogram,
                                    state_symbols=sub_symbols, count_matrix_full=self.count_matrix_full,
                                    state_histogram_full=self.state_histogram_full)

    def submodel_largest(self, connectivity_threshold: Union[str, float] = 0., directed: Optional[bool] = None,
                         probability_constraint: Optional[np.ndarray] = None, sort_by_population: bool = False):
        r"""
        Restricts this model to the submodel corresponding to the largest connected set of states after eliminating
        states that fall below the specified connectivity threshold.
        
        Parameters
        ----------
        connectivity_threshold : float or '1/n', optional, default=0.
            Connectivity threshold. counts that are below the specified value are disregarded when finding connected
            sets. In case of '1/n', the threshold gets resolved to :math:`1 / n\_states\_full`.
        directed : bool, optional, default=None
            Whether to look for connected sets in a directed graph or in an undirected one. Per default it looks whether
            a probability constraint is given. In case it is given it defaults to the undirected case, otherwise
            directed.
        probability_constraint : (N,) ndarray, optional, default=None
            Constraint on the whole state space (n_states_full). Only considers states that have positive probability.
        sort_by_population : bool, optional, default=False
            This flag can be used to use the connected set with the largest population.

        Returns
        -------
        submodel : TransitionCountModel
            The submodel.
        """
        if directed is None:
            # if probability constraint is given, we want undirected per default
            directed = probability_constraint is None
        if connectivity_threshold == '1/n':
            connectivity_threshold = 1. / self.n_states_full
        connectivity_threshold = float(connectivity_threshold)
        connected_sets = self.connected_sets(connectivity_threshold=connectivity_threshold, directed=directed,
                                             probability_constraint=probability_constraint,
                                             sort_by_population=sort_by_population)
        largest_connected_set = connected_sets[0]
        return self.submodel(largest_connected_set)

    def count_matrix_histogram(self) -> np.ndarray:
        r"""
        Computes a histogram over states represented in the count matrix. The magnitude of the values returned values
        depend on the mode which was used for counting.

        Returns
        -------
        A `(n_states,) np.ndarray` histogram over the collected counts per state.
        """
        return self.count_matrix.sum(axis=1)

    def is_connected(self, directed: bool = True) -> bool:
        r""" Dispatches to :meth:`tools.analysis.is_connected <deeptime.markov.tools.analysis.is_connected>`.
        """
        return is_connected(self.count_matrix, directed=directed)


class TransitionCountEstimator(EstimatorTransformer):
    r"""
    Estimator which produces a :class:`TransitionCountModel` given discretized trajectories.
    Hereby one can decide whether the count mode should be:

        * sample: A trajectory of length T will have :math:`T / \tau` counts at time indices

          .. math:: (0 \rightarrow \tau), (\tau \rightarrow 2 \tau), ..., (((T/ \tau )-1) \tau \rightarrow T)

        * sliding: A trajectory of length T will have :math:`T-\tau` counts at time indices

          .. math:: (0 \rightarrow \tau), (1 \rightarrow \tau+1), ..., (T-\tau-1 \rightarrow T-1)

          This introduces an overestimation of the actual count values by a factor of "lagtime". For
          maximum-likelihood MSMs this plays no role but it leads to wrong error bars in uncertainty estimation.

        * sliding-effective: See sliding mode, just that the resulting count matrix is divided by the lagtime after
          counting. This which can be shown to provide a likelihood that is the geometrical average
          over shifted subsamples of the trajectory, :math:`(s_1,\:s_{tau+1},\:...),\:(s_2,\:t_{tau+2},\:...),` etc.
          This geometrical average converges to the correct likelihood in the statistical limit
          :footcite:`trendelkamp2015estimation`.

        * effective: Uses an estimate of the transition counts that are statistically uncorrelated. Recommended
          when used with a Bayesian MSM. A description of the estimation procedure
          can be found in :footcite:`noe2015statistical`.

    Parameters
    ----------
    lagtime : int
        Distance between two frames in the discretized trajectories under which their potential change of state
        is considered a transition.
    count_mode : str
        One of "sample", "sliding", "sliding-effective", and "effective".

        * "sample" strides the trajectory with lagtime :math:`\tau` and uses the strided counts as transitions.
        * "sliding" uses a sliding window approach, yielding counts that are statistically correlated and too
          large by a factor of :math:`\tau`; in uncertainty estimation this yields wrong uncertainties.
        * "sliding-effective" takes "sliding" and divides it by :math:`\tau`, which can be shown to provide a
          likelihood that is the geometrical average over shifted subsamples of the trajectory,
          :math:`(s_1,\:s_{tau+1},\:...),\:(s_2,\:t_{tau+2},\:...),` etc. This geometrical average converges to
          the correct likelihood in the statistical limit :footcite:`trendelkamp2015estimation`.
        * "effective" uses an estimate of the transition counts that are statistically uncorrelated.
          Recommended when estimating Bayesian MSMs.
    n_states : int, optional, default=None
        Normally, the shape of the count matrix is a consequence of the number of encountered states in given
        discrete trajectories. However sometimes (for instance when scoring), only a portion of the discrete
        trajectories is passed but the count matrix should still have the correct shape. Then, this argument
        can be used to artificially set the number of states to the correct value.
    sparse : bool, optional, default=False
        Whether sparse matrices should be used for counting. This can make sense when the number of states is very
        large.

    See Also
    --------
    TransitionCountModel

    References
    ----------
    .. footbibliography::
    """

    def __init__(self, lagtime: int, count_mode: str, n_states=None, sparse=False):
        super().__init__()
        self.lagtime = lagtime
        self.count_mode = count_mode
        self.sparse = sparse
        self.n_states = n_states

    @property
    def count_mode(self):
        r""" The currently selected count mode. """
        return self._count_mode

    @count_mode.setter
    def count_mode(self, value):
        valids = ('sliding', 'sliding-effective', 'sample', 'effective')
        if value not in valids:
            raise ValueError("Invalid count mode \"{}\", possible values are {}.".format(value, valids))
        self._count_mode = value

    @property
    def sparse(self) -> bool:
        r""" Whether the resulting count matrix is stored in sparse or dense mode.

        :getter: Yields the currently configured sparsity setting.
        :setter: Sets whether to store count matrices sparsely.
        :type: bool
        """
        return self._sparse

    @sparse.setter
    def sparse(self, value: bool):
        self._sparse = bool(value)

    @property
    def n_states(self) -> Optional[bool]:
        r""" The number of states in discrete trajectories. Can be used to override the effective shape
        of the resulting count matrix.

        :getter: Yields the currently set number of states or None.
        :setter: Sets the number of states to use or None.
        :type: bool or None
        """
        return self._n_states

    @n_states.setter
    def n_states(self, value):
        self._n_states = value

    @property
    def lagtime(self) -> int:
        r""" The lagtime at which transitions are counted. """
        return self._lagtime

    @lagtime.setter
    def lagtime(self, value: int):
        self._lagtime = value

    def fetch_model(self) -> Optional[TransitionCountModel]:
        r"""
        Yields the latest estimated :class:`TransitionCountModel`. Might be None if fetched before any data was fit.

        Returns
        -------
        The latest :class:`TransitionCountModel` or None.
        """
        return self._model

    def fit(self, data, *args, **kw):
        r""" Counts transitions at given lag time according to configuration of the estimator.

        Parameters
        ----------
        data : array_like or list of array_like
            discretized trajectories
        """
        from deeptime.markov import count_states
        dtrajs = ensure_dtraj_list(data)

        # basic count statistics
        histogram = count_states(dtrajs, ignore_negative=True)

        # Compute count matrix
        count_mode = self.count_mode
        lagtime = self.lagtime
        count_matrix = TransitionCountEstimator.count(count_mode, dtrajs, lagtime, sparse=self.sparse,
                                                      n_jobs=kw.pop('n_jobs', None))
        if self.n_states is not None and self.n_states > count_matrix.shape[0]:
            histogram = np.pad(histogram, pad_width=[(0, self.n_states - count_matrix.shape[0])])
            if issparse(count_matrix):
                count_matrix = scipy.sparse.csr_matrix((count_matrix.data, count_matrix.indices, count_matrix.indptr),
                                                       shape=(self.n_states, self.n_states))
            else:
                n_pad = self.n_states - count_matrix.shape[0]
                count_matrix = np.pad(count_matrix, pad_width=[(0, n_pad), (0, n_pad)])

        # initially state symbols, full count matrix, and full histogram can be left None because they coincide
        # with the input arguments
        self._model = TransitionCountModel(
            count_matrix=count_matrix, counting_mode=count_mode, lagtime=lagtime, state_histogram=histogram
        )
        return self

    @staticmethod
    def count(count_mode: str, dtrajs: List[np.ndarray], lagtime: int, sparse: bool = False, n_jobs=None):
        r""" Computes a count matrix based on a counting mode, some discrete trajectories, a lagtime, and
        whether to use sparse matrices.

        Parameters
        ----------
        count_mode : str
            The counting mode to use. One of "sample", "sliding", "sliding-effective", and "effective".
            See :meth:`__init__` for a more detailed description.
        dtrajs : array_like or list of array_like
            Discrete trajectories, i.e., a list of arrays which contain non-negative integer values. A single ndarray
            can also be passed, which is then treated as if it was a list with that one ndarray in it.
        lagtime : int
            Distance between two frames in the discretized trajectories under which their potential change of state
            is considered a transition.
        sparse : bool, default=False
            Whether to use sparse matrices or dense matrices. Sparse matrices can make sense when dealing with a lot of
            states.
        n_jobs : int, optional, default=None
            This only has an effect in effective counting. Determines the number of cores to use for estimating
            statistical inefficiencies. Default resolves to number of available cores.

        Returns
        -------
        count_matrix : (N, N) ndarray or sparse array
            The computed count matrix. Can be ndarray or sparse depending on whether sparse was set to true or false.
            N is the number of encountered states, i.e., :code:`np.max(dtrajs)+1`.

        Example
        -------
        >>> dtrajs = [np.array([0,0,1,1]), np.array([0,0,1])]
        >>> count_matrix = TransitionCountEstimator.count(
        ...     count_mode="sliding", dtrajs=dtrajs, lagtime=1, sparse=False
        ... )
        >>> np.testing.assert_equal(count_matrix, np.array([[2, 2], [0, 1]]))
        """
        if count_mode == 'sliding' or count_mode == 'sliding-effective':
            count_matrix = msmest.count_matrix(dtrajs, lagtime, sliding=True, sparse_return=sparse)
            if count_mode == 'sliding-effective':
                count_matrix /= lagtime
        elif count_mode == 'sample':
            count_matrix = msmest.count_matrix(dtrajs, lagtime, sliding=False, sparse_return=sparse)
        elif count_mode == 'effective':
            count_matrix = msmest.effective_count_matrix(dtrajs, lagtime, n_jobs=n_jobs)
            if not sparse and issparse(count_matrix):
                count_matrix = count_matrix.toarray()
        else:
            raise ValueError('Count mode {} is unknown.'.format(count_mode))
        return count_matrix
