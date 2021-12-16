import numpy as np

from deeptime.util.types import ensure_dtraj_list


def map_dtrajs_to_symbols(dtrajs, state_symbols: np.ndarray, n_states_full: int,
                          empty_symbol: np.int32 = -1, check=False):
    r"""A list of integer arrays with the discrete trajectories mapped to the currently used set of symbols.

    Parameters
    ----------
    dtrajs : array_like or list of array_like
        discretized trajectories
    state_symbols : ndarray
        the state symbols to restrict to
    n_states_full : int
        Total number of states.
    empty_symbol: np.int32, optional, default=-1
        The artificial state that is mapped to, if it is not contained in state_symbols.
    check : bool, default=False
        Whether to convert the input dtrajs to list of dtrajs or assume it is a list of dtrajs already.

    Returns
    -------
    transformed_dtrajs : List[np.ndarray]
        Mapped dtrajs.
    """
    if check:
        dtrajs = ensure_dtraj_list(dtrajs)
    mapping = np.full(n_states_full, empty_symbol, dtype=np.int32)
    mapping[state_symbols] = np.arange(len(state_symbols))
    return [mapping[dtraj] for dtraj in dtrajs]


class DiscreteStatesManager:
    r"""Class that manages discrete states. Resolves mapping between state symbols and states;
        contains functions for mapping to a subset (e.g. an active (connected) set of an MSM).

       Parameters
       ----------
       state_symbols : ndarray
            array that contains the integer state symbols used in a model
        n_states_full : int
            number of full states in the modeled state space
        blank_state : int, optional, default=-1
            definition of the blank or empty state used to map observations to that are not part of the set of
            state symbols.

       """
    def __init__(self, state_symbols: np.ndarray, n_states_full: int, blank_state=-1):
        self._blank_state = blank_state
        self._state_symbols = state_symbols
        self._n_states_full = n_states_full
        self._state_symbols_with_blank = np.concatenate((state_symbols, (blank_state,)))

        if self.n_states_full < self.n_states:
            # full number of states must be at least as large as n_states
            raise ValueError(f"Number of states was bigger than full number of "
                             f"states. (#states = {self.n_states}, #states_full = {self.n_states_full}).")

    @property
    def n_states(self):
        r"""
        Property to determine the number of observed states or state symbols that are part of the active set.

        Returns
        -------
        Number of observed states in active set
        """
        return len(self.state_symbols)

    @property
    def n_states_full(self):
        r"""
        Property to determine the number of observed states or state symbols in all data.

        Returns
        -------
        Number of observed states
        """
        return self._n_states_full

    @property
    def states(self) -> np.ndarray:
        r""" The states in this model, i.e., a iota range from 0 (inclusive) to :meth:`n_states` (exclusive).
        See also: :meth:`state_symbols`. """
        return np.arange(self.n_states)

    @property
    def state_symbols(self) -> np.ndarray:
        r""" Symbols for states that are represented in this count model. """
        return self._state_symbols_with_blank[:-1]

    @property
    def state_symbols_with_blank(self):
        r""" Symbols for states that are represented in this count model plus a state `blank` for states which are
        not represented in this count model. """
        return self._state_symbols_with_blank

    @property
    def blank_state(self):
        r"""
        Property that holds the blank state, i.e. the symbol that denotes an inactive state.

        Returns
        -------
        blank state symbol
        """
        return self._blank_state

    @property
    def is_full(self) -> bool:
        r""" Determine whether this counting model refers to the full model that represents all states of the data.

        Returns
        -------
        is_full_model : bool
            Whether this counting model represents all states of the data.
        """
        return self.n_states == self.n_states_full

    @property
    def selected_state_fraction(self) -> float:
        """The represented fraction of states."""
        return float(self.n_states) / float(self.n_states_full)

    def project(self, dtrajs, check=False):
        r"""A list of integer arrays with the discrete trajectories mapped to the currently used set of symbols.
        For example, if there has been a subselection of the model for connectivity='largest', the indices will be
        given within the connected set, frames that do not correspond to a considered symbol are set to -1.

        Parameters
        ----------
        dtrajs : array_like or list of array_like
            Discretized trajectories.
        check : bool, optional, default=False
            Whether to check the dtrajs input for validity.

        Returns
        -------
        array_like or list of array_like
            Curated discretized trajectories so that unconsidered symbols are mapped to -1.
        """

        if self.is_full:
            # no-op
            return dtrajs
        else:
            return map_dtrajs_to_symbols(dtrajs, self.state_symbols, self.n_states_full, check=check)

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
        return self.state_symbols_with_blank[states]

    def symbols_to_states(self, symbols):
        r"""
        Converts a set of symbols to state indices. The symbols which
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

    def subselect_states(self, states: np.ndarray):
        r"""
        Creates a new instance of a DiscreteStatesManager with a subset of states
        Parameters
        ----------
        states : ndarray
            the selection of states to create a new DiscreteStatesManager for

        Returns
        -------
        DiscreteStatesManager
            Discrete states manager for subselection
        """
        states = np.atleast_1d(states)
        sub_symbols = self.state_symbols[states]
        return DiscreteStatesManager(sub_symbols, self.n_states_full, self.blank_state)

    def __len__(self):
        return len(self.state_symbols)
