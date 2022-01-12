import unittest

from deeptime.markov import DiscreteStatesManager
import numpy as np

from numpy.testing import assert_equal, assert_array_equal, assert_


class TestDiscreteStateManagerIdentityMap(unittest.TestCase):
    r""" Checks if DiscreteStatesManager maps states to symbols in a case where both are the same.
    """

    def setUp(self) -> None:
        self.dtraj = np.array([0, 0, 0, 1, 0, 1, 1])
        self.state_symbols = np.array([0, 1])
        self.states = self.state_symbols.copy()
        self.n_states_full = 2
        self.dsm = DiscreteStatesManager(self.state_symbols, self.n_states_full)

    def test_statenumbers(self):
        assert_equal(self.dsm.n_states, self.n_states_full)
        assert_equal(self.dsm.n_states_full, self.n_states_full)

    def test_states_symbols(self):
        assert_(self.dsm.is_full)
        assert_array_equal(self.dsm.states, self.states)
        assert_array_equal(self.dsm.state_symbols, self.state_symbols)

    def test_mappings(self):
        assert_array_equal(self.dsm.project(self.dtraj), self.dtraj)
        assert_array_equal(self.dsm.states_to_symbols(self.states), self.state_symbols)
        assert_array_equal(self.dsm.symbols_to_states(self.state_symbols), self.states)

    def test_subselect(self):
        new_dsm = self.dsm.subselect_states(np.array([0]))
        assert_equal(len(new_dsm), 1)


class TestDiscreteStateManager(unittest.TestCase):
    """ Checks if DiscreteStatesManager maps states to symbols for a case of non-contiguous symbols.
        Also tests if sub-selecting states reproduces expectable results.
    """

    def setUp(self) -> None:
        self.dtraj = np.array([0, 0, 0, 5, 0, 5, 5, 10, 10])
        self.state_symbols = np.array([0, 5, 10])
        self.states = np.array([0, 1, 2])
        self.n_states_full = 25
        self.dsm = DiscreteStatesManager(self.state_symbols, self.n_states_full)

        self.sel_states = np.array([0, 1])
        self.sel_state_symbols = np.array([0, 5])
        self.sel_dsm = self.dsm.subselect_states(self.sel_states)

    def test_statenumbers(self):
        assert_(not self.dsm.is_full)
        assert_equal(self.dsm.n_states, len(self.states))

    def test_states_symbols(self):
        assert_array_equal(self.dsm.states, self.states)
        assert_array_equal(self.dsm.state_symbols, self.state_symbols)

    def test_mappings(self):
        projected_dtraj = (self.dtraj/5).astype(int)
        assert_array_equal(self.dsm.project(self.dtraj), projected_dtraj)
        assert_array_equal(self.dsm.states_to_symbols(self.states), self.state_symbols)
        assert_array_equal(self.dsm.symbols_to_states(self.state_symbols), self.states)

    def test_subselect_statenumbers(self):
        assert_array_equal(self.sel_dsm.states, self.sel_states)
        assert_array_equal(self.sel_dsm.state_symbols, self.sel_state_symbols)

    def test_subselect_states_symbols(self):
        assert_array_equal(self.sel_dsm.states, self.sel_states)
        assert_array_equal(self.sel_dsm.state_symbols, self.sel_state_symbols)

    def test_subselect_mappings(self):
        # tests only states contained in state selection
        projected_dtraj = (self.dtraj/5).astype(int)[:6]
        assert_array_equal(self.sel_dsm.project(self.dtraj[:6]), projected_dtraj[:6])
        assert_array_equal(self.sel_dsm.states_to_symbols(self.sel_states), self.sel_state_symbols)
        assert_array_equal(self.sel_dsm.symbols_to_states(self.sel_state_symbols), self.sel_states)

    def test_subselect_mappings_with_outliers(self):
        projected_dtraj = (self.dtraj/5).astype(int)
        projected_dtraj[projected_dtraj == 2] = -1
        assert_array_equal(self.sel_dsm.project(self.dtraj), projected_dtraj)
        assert_array_equal(self.sel_dsm.states_to_symbols(self.states), np.array([0, 5, -1]))
        # function discards states that are not in selection
        assert_array_equal(self.sel_dsm.symbols_to_states(self.state_symbols), [0, 1])
