import unittest

import numpy as np

from sktime.markovprocess import TransitionCountEstimator, Q_, TransitionCountModel
from tests.util import GenerateTestMatrix


class TestTransitionCountEstimator(unittest.TestCase):

    def test_properties(self):
        valid_count_modes = "sample", "sliding", "sliding-effective", "effective"
        for mode in valid_count_modes:
            estimator = TransitionCountEstimator(lagtime=5, count_mode=mode, physical_time="10 ns")
            self.assertEqual(estimator.count_mode, mode)
            np.testing.assert_equal(estimator.lagtime, 5)
            assert Q_("10 ns") == estimator.physical_time, \
                "expected 10 ns as physical time but got {}".format(estimator.physical_time)

    def test_sample_counting(self):
        dtraj = np.array([0, 0, 0, 0, 1, 1, 0, 1])
        estimator = TransitionCountEstimator(lagtime=2, count_mode="sample")
        model = estimator.fit(dtraj).fetch_model()
        # sample strides the trajectory with "lag" and then counts instantaneous transitions
        # get counts 0 -> 0, 0 -> 1, 1 -> 0
        np.testing.assert_array_equal(model.count_matrix.toarray(), np.array([[1., 1.], [1., 0.]]))
        np.testing.assert_equal(model.lagtime, 2)
        assert model.counting_mode == "sample", "expected sample counting mode, got {}".format(model.counting_mode)
        assert Q_("1 step") == model.physical_time, "no physical time specified, expecting 'step' " \
                                                    "but got {}".format(model.physical_time)
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
        np.testing.assert_array_equal(model.count_matrix.toarray(), np.array([[2., 2.], [1., 1.]]))
        np.testing.assert_equal(model.lagtime, 2)
        assert model.counting_mode == "sliding", "expected sliding counting mode, got {}".format(model.counting_mode)
        assert Q_("1 step") == model.physical_time, "no physical time specified, expecting 'step' " \
                                                    "but got {}".format(model.physical_time)
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
        np.testing.assert_array_equal(model.count_matrix.toarray(), np.array([[2., 2.], [1., 1.]]) / 2.)
        np.testing.assert_equal(model.lagtime, 2)
        assert model.counting_mode == "sliding-effective", \
            "expected sliding-effective counting mode, got {}".format(model.counting_mode)
        assert Q_("1 step") == model.physical_time, "no physical time specified, expecting 'step' " \
                                                    "but got {}".format(model.physical_time)
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
        # todo actually compute this and see if it makes sense
        np.testing.assert_array_equal(model.count_matrix.toarray(), np.array([[1.6, 1.6], [1., 1.]]))
        np.testing.assert_equal(model.lagtime, 2)
        assert model.counting_mode == "effective", "expected effective counting mode, " \
                                                   "got {}".format(model.counting_mode)
        assert Q_("1 step") == model.physical_time, "no physical time specified, expecting 'step' " \
                                                    "but got {}".format(model.physical_time)
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
        np.testing.assert_equal(model.physical_time, Q_("1 step"))
        np.testing.assert_equal(model.count_matrix_full, count_matrix)
        np.testing.assert_equal(model.counting_mode, "sliding")


    def _test_submodel(self, histogram):
        # three connected components: ((1, 2), (0), (3))
        count_matrix = np.array([[10., 0., 0., 0.], [0., 1., 1., 0.], [0., 1., 1., 0.], [0., 0., 0., 1]])
        model = TransitionCountModel(count_matrix, counting_mode="sliding", state_histogram=histogram)

        self._check_submodel_transitive_properties(histogram, count_matrix, model)

        if histogram is not None:
            np.testing.assert_equal(model.selected_count_fraction, 1.)
            np.testing.assert_equal(model.total_count, 100 + 10 + 10 + 10)
            np.testing.assert_equal(model.visited_set, [0, 1, 2, 3])
        else:
            np.testing.assert_raises(RuntimeError, model.selected_count_fraction)
            np.testing.assert_raises(RuntimeError, model.total_count)
            np.testing.assert_raises(RuntimeError, model.visited_set)

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
            np.testing.assert_raises(RuntimeError, submodel.selected_count_fraction)
            np.testing.assert_raises(RuntimeError, submodel.total_count)
            np.testing.assert_raises(RuntimeError, submodel.visited_set)
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
            np.testing.assert_raises(RuntimeError, subsubmodel.selected_count_fraction)
            np.testing.assert_raises(RuntimeError, subsubmodel.total_count)
            np.testing.assert_raises(RuntimeError, subsubmodel.visited_set)
        np.testing.assert_equal(subsubmodel.count_matrix, np.array([[1]]))
        np.testing.assert_equal(subsubmodel.selected_state_fraction, 0.25)
        sets = subsubmodel.connected_sets(connectivity_threshold=0, directed=True, probability_constraint=None)
        np.testing.assert_equal(len(sets), 1)
        np.testing.assert_equal(len(sets[0]), 1)
        assert 0 in sets[0], "state 0 should be in the connected set, but got {}".format(sets[0])
        np.testing.assert_equal(subsubmodel.state_symbols, [2])
        np.testing.assert_(not subsubmodel.is_full_model)
        np.testing.assert_equal(subsubmodel.n_states, 1)


if __name__ == '__main__':
    unittest.main()
