import unittest

import numpy as np

from deeptime.markov import TransitionCountEstimator, TransitionCountModel
from tests.testing_utilities import GenerateTestMatrix


class TestTransitionCountEstimator(unittest.TestCase):

    def test_properties(self):
        valid_count_modes = "sample", "sliding", "sliding-effective", "effective"
        for mode in valid_count_modes:
            estimator = TransitionCountEstimator(lagtime=5, count_mode=mode)
            self.assertEqual(estimator.count_mode, mode)
            np.testing.assert_equal(estimator.lagtime, 5)

    def test_sample_counting(self):
        dtraj = np.array([0, 0, 0, 0, 1, 1, 0, 1])
        estimator = TransitionCountEstimator(lagtime=2, count_mode="sample")
        model = estimator.fit(dtraj).fetch_model()
        # sample strides the trajectory with "lag" and then counts instantaneous transitions
        # get counts 0 -> 0, 0 -> 1, 1 -> 0
        np.testing.assert_array_equal(model.count_matrix, np.array([[1., 1.], [1., 0.]]))
        np.testing.assert_equal(model.lagtime, 2)
        assert model.counting_mode == "sample", "expected sample counting mode, got {}".format(model.counting_mode)

        np.testing.assert_equal(model.state_symbols, [0, 1], err_msg="Trajectory only contained states 0 and 1")
        np.testing.assert_equal(model.n_states, 2)
        np.testing.assert_equal(model.state_histogram, [5, 3])
        assert model.is_full_model
        np.testing.assert_equal(model.selected_count_fraction, 1)
        np.testing.assert_equal(model.selected_state_fraction, 1)
        np.testing.assert_equal(model.total_count, len(dtraj))
        np.testing.assert_equal(model.visited_set, [0, 1])

    def test_sliding_counting(self):
        dtraj = np.array([0, 0, 0, 0, 1, 1, 0, 1])
        estimator = TransitionCountEstimator(lagtime=2, count_mode="sliding")
        model = estimator.fit(dtraj).fetch_model()
        # sliding window across trajectory counting transitions, overestimating total count:
        # 0 -> 0, 0 -> 0, 0 -> 1, 0-> 1, 1-> 0, 1-> 1
        np.testing.assert_array_equal(model.count_matrix, np.array([[2., 2.], [1., 1.]]))
        np.testing.assert_equal(model.lagtime, 2)
        assert model.counting_mode == "sliding", "expected sliding counting mode, got {}".format(model.counting_mode)
        np.testing.assert_equal(model.state_symbols, [0, 1], err_msg="Trajectory only contained states 0 and 1")
        np.testing.assert_equal(model.n_states, 2)
        np.testing.assert_equal(model.state_histogram, [5, 3])
        assert model.is_full_model
        np.testing.assert_equal(model.selected_count_fraction, 1)
        np.testing.assert_equal(model.selected_state_fraction, 1)
        np.testing.assert_equal(model.total_count, len(dtraj))
        np.testing.assert_equal(model.visited_set, [0, 1])

    def test_sliding_effective_counting(self):
        dtraj = np.array([0, 0, 0, 0, 1, 1, 0, 1])
        estimator = TransitionCountEstimator(lagtime=2, count_mode="sliding-effective")
        model = estimator.fit(dtraj).fetch_model()
        # sliding window across trajectory counting transitions, overestimating total count:
        # 0 -> 0, 0 -> 0, 0 -> 1, 0-> 1, 1-> 0, 1-> 1
        # then divide by lagtime
        np.testing.assert_array_equal(model.count_matrix, np.array([[2., 2.], [1., 1.]]) / 2.)
        np.testing.assert_equal(model.lagtime, 2)
        assert model.counting_mode == "sliding-effective", \
            "expected sliding-effective counting mode, got {}".format(model.counting_mode)
        np.testing.assert_equal(model.state_symbols, [0, 1], err_msg="Trajectory only contained states 0 and 1")
        np.testing.assert_equal(model.n_states, 2)
        np.testing.assert_equal(model.state_histogram, [5, 3])
        assert model.is_full_model
        np.testing.assert_equal(model.selected_count_fraction, 1)
        np.testing.assert_equal(model.selected_state_fraction, 1)
        np.testing.assert_equal(model.total_count, len(dtraj))
        np.testing.assert_equal(model.visited_set, [0, 1])

    def test_effective_counting(self):
        dtraj = np.array([0, 0, 0, 0, 1, 1, 0, 1])
        estimator = TransitionCountEstimator(lagtime=2, count_mode="effective")
        model = estimator.fit(dtraj).fetch_model()
        # effective counting
        np.testing.assert_array_equal(model.count_matrix, np.array([[1.6, 1.6], [1., 1.]]))
        np.testing.assert_equal(model.lagtime, 2)
        assert model.counting_mode == "effective", "expected effective counting mode, " \
                                                   "got {}".format(model.counting_mode)
        np.testing.assert_equal(model.state_symbols, [0, 1], err_msg="Trajectory only contained states 0 and 1")
        np.testing.assert_equal(model.n_states, 2)
        np.testing.assert_equal(model.state_histogram, [5, 3])
        assert model.is_full_model
        np.testing.assert_equal(model.selected_count_fraction, 1)
        np.testing.assert_equal(model.selected_state_fraction, 1)
        np.testing.assert_equal(model.total_count, len(dtraj))
        np.testing.assert_equal(model.visited_set, [0, 1])


class TestTransitionCountModel(unittest.TestCase, metaclass=GenerateTestMatrix):
    params = {
        '_test_submodel': [dict(histogram=hist) for hist in [None, np.array([100, 10, 10, 10])]]
    }

    @staticmethod
    def _check_submodel_transitive_properties(histogram, count_matrix, model: TransitionCountModel):
        """ checks properties of the model which do not / should not change when taking a submodel """
        np.testing.assert_equal(model.state_histogram_full, histogram)
        np.testing.assert_equal(model.lagtime, 1)
        np.testing.assert_equal(model.n_states_full, 4)
        np.testing.assert_equal(model.count_matrix_full, count_matrix)
        np.testing.assert_equal(model.counting_mode, "effective")

    def _test_submodel(self, histogram):
        # three connected components: ((1, 2), (0), (3))
        count_matrix = np.array([[10., 0., 0., 0.], [0., 1., 1., 0.], [0., 1., 1., 0.], [0., 0., 0., 1]])
        model = TransitionCountModel(count_matrix, counting_mode="effective", state_histogram=histogram)

        self._check_submodel_transitive_properties(histogram, count_matrix, model)

        if histogram is not None:
            np.testing.assert_equal(model.selected_count_fraction, 1.)
            np.testing.assert_equal(model.total_count, 100 + 10 + 10 + 10)
            np.testing.assert_equal(model.visited_set, [0, 1, 2, 3])
        else:
            with np.testing.assert_raises(RuntimeError):
                print(model.selected_count_fraction)
            with np.testing.assert_raises(RuntimeError):
                print(model.total_count)
            with np.testing.assert_raises(RuntimeError):
                print(model.visited_set)

        np.testing.assert_equal(model.count_matrix, count_matrix)
        np.testing.assert_equal(model.selected_state_fraction, 1.)

        sets = model.connected_sets(connectivity_threshold=0, directed=True, probability_constraint=None)
        np.testing.assert_equal(len(sets), 3)
        np.testing.assert_equal(len(sets[0]), 2)
        np.testing.assert_equal(len(sets[1]), 1)
        np.testing.assert_equal(len(sets[2]), 1)
        np.testing.assert_equal(model.state_symbols, [0, 1, 2, 3])
        np.testing.assert_(model.is_full_model)
        np.testing.assert_equal(model.state_histogram, histogram)
        np.testing.assert_equal(model.n_states, 4)
        assert 1 in sets[0] and 2 in sets[0], "expected states 1 and 2 in largest connected set, got {}".format(sets[0])

        submodel = model.submodel(sets[0])
        self._check_submodel_transitive_properties(histogram, count_matrix, submodel)
        if histogram is not None:
            np.testing.assert_equal(submodel.state_histogram, [10, 10])
            np.testing.assert_equal(submodel.selected_count_fraction, 20. / 130.)
            np.testing.assert_equal(submodel.total_count, 20)
            np.testing.assert_equal(submodel.visited_set, [0, 1])
        else:
            np.testing.assert_equal(submodel.state_histogram, None)
            with np.testing.assert_raises(RuntimeError):
                print(submodel.selected_count_fraction)
            with np.testing.assert_raises(RuntimeError):
                print(submodel.total_count)
            with np.testing.assert_raises(RuntimeError):
                print(submodel.visited_set)
        np.testing.assert_equal(submodel.count_matrix, np.array([[1, 1], [1, 1]]))
        np.testing.assert_equal(submodel.selected_state_fraction, 0.5)
        sets = submodel.connected_sets(connectivity_threshold=0, directed=True, probability_constraint=None)
        np.testing.assert_equal(len(sets), 1)
        np.testing.assert_equal(len(sets[0]), 2)
        assert 0 in sets[0] and 1 in sets[0], "states 0 and 1 should be in the connected set, " \
                                              "but got {}".format(sets[0])
        np.testing.assert_equal(submodel.state_symbols, [1, 2])
        np.testing.assert_(not submodel.is_full_model)
        np.testing.assert_equal(submodel.n_states, 2)

        subsubmodel = submodel.submodel([1])
        self._check_submodel_transitive_properties(histogram, count_matrix, subsubmodel)
        if histogram is not None:
            np.testing.assert_equal(subsubmodel.state_histogram, [10])
            np.testing.assert_equal(subsubmodel.selected_count_fraction, 10. / 130.)
            np.testing.assert_equal(subsubmodel.total_count, 10)
            np.testing.assert_equal(subsubmodel.visited_set, [0])
        else:
            np.testing.assert_equal(subsubmodel.state_histogram, None)
            with np.testing.assert_raises(RuntimeError):
                print(subsubmodel.selected_count_fraction)
            with np.testing.assert_raises(RuntimeError):
                print(subsubmodel.total_count)
            with np.testing.assert_raises(RuntimeError):
                print(subsubmodel.visited_set)
        np.testing.assert_equal(subsubmodel.count_matrix, np.array([[1]]))
        np.testing.assert_equal(subsubmodel.selected_state_fraction, 0.25)
        sets = subsubmodel.connected_sets(connectivity_threshold=0, directed=True, probability_constraint=None)
        np.testing.assert_equal(len(sets), 1)
        np.testing.assert_equal(len(sets[0]), 1)
        assert 0 in sets[0], "state 0 should be in the connected set, but got {}".format(sets[0])
        np.testing.assert_equal(subsubmodel.state_symbols, [2])
        np.testing.assert_(not subsubmodel.is_full_model)
        np.testing.assert_equal(subsubmodel.n_states, 1)

    def test_symbols_to_states_conversion(self):
        dtraj = np.array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1])
        model = TransitionCountEstimator(lagtime=1, count_mode='sliding').fit(dtraj).fetch_model()
        submodel = model.submodel([2, 3, 4, 6])
        np.testing.assert_equal(submodel.states_to_symbols(submodel.states), [2, 3, 4, 6])
        np.testing.assert_array_equal(submodel.state_symbols, [2, 3, 4, 6])
        states = submodel.symbols_to_states([6, 2, 0])
        for state in states:
            # check that each returned state points to one of the requested symbols if still represented in model
            # (symbol 0 is not contained in submodel)
            np.testing.assert_(submodel.state_symbols[state] in (6, 2))
        np.testing.assert_(0 not in submodel.state_symbols)
