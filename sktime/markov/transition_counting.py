from functools import wraps
from typing import Union, Optional, List, Callable

import numpy as np
import scipy
from msmtools import estimation as msmest
from scipy.sparse import coo_matrix, issparse

from sktime.base import Estimator, Model
from sktime.markov import Q_
from sktime.markov.util import count_states, compute_connected_sets
from sktime.util import submatrix, ensure_dtraj_list

__author__ = 'noe, clonker'


def requires_state_histogram(func : Callable) -> Callable:
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

    See Also
    --------
    TransitionCountEstimator : estimator which can produce this kind of model
    """

    def __init__(self, count_matrix, counting_mode: Optional[str] = None,
                 lagtime: int = 1, state_histogram: Optional[np.ndarray] = None,
                 physical_time='1 step',
                 state_symbols: Optional[np.ndarray] = None,
                 count_matrix_full=None,
                 state_histogram_full: Optional[np.ndarray] = None):
        r"""Creates a new TransitionCountModel. This can be used to, e.g., construct Markov state models. The minimal
        requirement for instantiation is a count matrix, but statistics of the data can also be provided.

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
        physical_time : Unit or str, default='step'
            Description of the physical time unit corresponding to one time step of the
            transitioning process (aka lag time). May be used by analysis methods such as plotting
            tools to pretty-print the axes.
            By default 'step', i.e. there is no physical time unit. Permitted units are

            *  'fs',  'femtosecond'
            *  'ps',  'picosecond'
            *  'ns',  'nanosecond'
            *  'us',  'microsecond'
            *  'ms',  'millisecond'
            *  's',   'second'
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

        if count_matrix is None:
            raise ValueError("count matrix was None")

        self._count_matrix = count_matrix
        self._counting_mode = counting_mode
        self._lag = lagtime
        self._physical_time = Q_(physical_time) if isinstance(physical_time, (str, int)) else physical_time
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
    def physical_time(self) -> Q_:
        """Time interval between discrete steps of the time series."""
        return self._physical_time

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
    def count_matrix_full(self) -> np.ndarray:
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
                                    state_symbols=sub_symbols, physical_time=self.physical_time,
                                    count_matrix_full=self.count_matrix_full,
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


class TransitionCountEstimator(Estimator):
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
          This geometrical average converges to the correct likelihood in the statistical limit [1]_. "effective"
          uses an estimate of the transition counts that are statistically uncorrelated.

        * effective: Uses an estimate of the transition counts that are statistically uncorrelated. Recommended
          when used with a Bayesian MSM.

    See Also
    --------
    TransitionCountModel : type of model this estimator produces

    References
    ----------

    .. [1] Trendelkamp-Schroer B, H Wu, F Paul and F Noe. 2015:
       Reversible Markov models of molecular kinetics: Estimation and uncertainty.
       J. Chem. Phys. 143, 174101 (2015); https://doi.org/10.1063/1.4934536
    """

    def __init__(self, lagtime: int, count_mode: str, physical_time='1 step', sparse=False):
        r"""Constructs a transition count estimator that can be used to estimate :class:`TransitionCountModel` s.

        Parameters
        ----------
        lagtime : int
            Distance between two frames in the discretized trajectories under which their potential change of state
            is considered a transition.
        count_mode : str
            one of "sample", "sliding", "sliding-effective", and "effective".

            * "sample" strides the trajectory with lagtime :math:`\tau` and uses the strided counts as transitions.
            * "sliding" uses a sliding window approach, yielding counts that are statistically correlated and too
              large by a factor of :math:`\tau`; in uncertainty estimation this yields wrong uncertainties.
            * "sliding-effective" takes "sliding" and divides it by :math:`\tau`, which can be shown to provide a
              likelihood that is the geometrical average over shifted subsamples of the trajectory,
              :math:`(s_1,\:s_{tau+1},\:...),\:(s_2,\:t_{tau+2},\:...),` etc. This geometrical average converges to
              the correct likelihood in the statistical limit [1]_.
            * "effective" uses an estimate of the transition counts that are statistically uncorrelated.
              Recommended when estimating Bayesian MSMs.
        physical_time : str, optional, default='1 step'
            Description of the physical time of the input trajectories. May be used
            by analysis algorithms such as plotting tools to pretty-print the axes.
            By default '1 step', i.e. there is no physical time unit. Specify by a
            number, whitespace and unit. Permitted units are (* is an arbitrary
            string):

            |  'fs',  'femtosecond*'
            |  'ps',  'picosecond*'
            |  'ns',  'nanosecond*'
            |  'us',  'microsecond*'
            |  'ms',  'millisecond*'
            |  's',   'second*'
        """
        super().__init__()
        self.lagtime = lagtime
        self.count_mode = count_mode
        self.physical_time = physical_time
        self.sparse = sparse

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
    def lagtime(self) -> int:
        r""" The lagtime at which transitions are counted. """
        return self._lagtime

    @lagtime.setter
    def lagtime(self, value: int):
        self._lagtime = value

    @property
    def physical_time(self) -> Q_:
        r"""
        A description of the physical time for input trajectories. Specify by a number, whitespace, and unit.
        Permitted units are 'fs', 'ps', 'ns', 'us', 'ms', 's', and 'step'.

        :getter: Yields a :class:`pint.Quantity` describing the physical time.
        :setter: Sets a new physical time. Strings describing units are also accepted.
        :type: pint.Quantity
        """
        return self._physical_time

    @physical_time.setter
    def physical_time(self, value: str):
        self._physical_time = Q_(value)

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
        dtrajs = ensure_dtraj_list(data)

        # basic count statistics
        histogram = count_states(dtrajs, ignore_negative=True)

        # Compute count matrix
        count_mode = self.count_mode
        lagtime = self.lagtime
        if count_mode == 'sliding' or count_mode == 'sliding-effective':
            count_matrix = msmest.count_matrix(dtrajs, lagtime, sliding=True, sparse_return=self.sparse)
            if count_mode == 'sliding-effective':
                count_matrix /= lagtime
        elif count_mode == 'sample':
            count_matrix = msmest.count_matrix(dtrajs, lagtime, sliding=False, sparse_return=self.sparse)
        elif count_mode == 'effective':
            count_matrix = msmest.effective_count_matrix(dtrajs, lagtime)
            if not self.sparse and issparse(count_matrix):
                count_matrix = count_matrix.toarray()
        else:
            raise ValueError('Count mode {} is unknown.'.format(count_mode))

        # initially state symbols, full count matrix, and full histogram can be left None because they coincide
        # with the input arguments
        self._model = TransitionCountModel(
            count_matrix=count_matrix, counting_mode=count_mode, lagtime=lagtime, state_histogram=histogram,
            physical_time=self.physical_time
        )

        return self
