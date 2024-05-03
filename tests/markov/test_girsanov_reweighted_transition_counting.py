import unittest

import numpy as np
from numpy.testing import assert_equal, assert_array_equal

from deeptime.markov import TransitionCountEstimator, GirsanovReweightingEstimator

class TestGirsanovReweightingEstimator(unittest.TestCase):

    def test_properties(self):
        valid_count_modes = "sample", "sliding", "sliding-effective", "effective"
        for mode in valid_count_modes:
            estimator = GirsanovReweightingEstimator(lagtime=5, count_mode=mode)
            self.assertEqual(estimator.count_mode, mode)
            assert_equal(estimator.lagtime, 5)

    def test_sample_counting(self):
        dtraj = np.array([0, 0, 0, 0, 1, 1, 0, 1])
        _reweighting = (np.array([1, 1, 1, 1, 1, 1, 1, 1]),np.array([1, 1, 1, 1, 1, 1, 1, 1]))
        estimator = GirsanovReweightingEstimator(lagtime=2, count_mode="sample")
        with self.assertRaises(ValueError):
            estimator.fit(dtraj[:-1], reweighting_factors=_reweighting).fetch_model()

    def test_sliding_counting(self):
        dtraj = np.array([0, 0, 0, 0, 1, 1, 0, 1])
        _reweighting = (np.array([1, 1, 1, 1, 1, 1, 1, 1]),np.array([1, 1, 1, 1, 1, 1, 1, 1]))
        rwght_estimator = GirsanovReweightingEstimator(lagtime=2, count_mode="sliding")
        rwght_model = rwght_estimator.fit(dtraj,reweighting_factors=_reweighting).fetch_model()
        estimator = TransitionCountEstimator(lagtime=2, count_mode="sliding")
        model = estimator.fit(dtraj).fetch_model()
        # sliding window across trajectory counting transitions, overestimating total count:
        # 0 -> 0, 0 -> 0, 0 -> 1, 0-> 1, 1-> 0, 1-> 1
        # relative values to compare with unweighted count matrix -> np.array([[2.,2.],[1.,1.]])/2.
        assert_array_equal(rwght_model.count_matrix/np.max(rwght_model.count_matrix), model.count_matrix/np.max(model.count_matrix))
        assert_equal(rwght_model.lagtime, 2)
        assert rwght_model.counting_mode == "sliding", "expected sliding counting mode, got {}".format(rwght_model.counting_mode)
        assert_equal(rwght_model.state_symbols, [0, 1], err_msg="Trajectory only contained states 0 and 1")
        assert_equal(rwght_model.n_states, 2)
        assert_equal(rwght_model.state_histogram, [5, 3])
        assert rwght_model.is_full_model
        assert_equal(rwght_model.selected_count_fraction, 1)
        assert_equal(rwght_model.selected_state_fraction, 1)
        assert_equal(rwght_model.total_count, len(dtraj))
        assert_equal(rwght_model.visited_set, [0, 1])

    def test_sliding_effective_counting(self):
        dtraj = np.array([0, 0, 0, 0, 1, 1, 0, 1])
        _reweighting = (np.array([1, 1, 1, 1, 1, 1, 1, 1]),np.array([1, 1, 1, 1, 1, 1, 1, 1]))
        estimator = GirsanovReweightingEstimator(lagtime=2, count_mode="sliding-effective")
        with self.assertRaises(ValueError):
            estimator.fit(dtraj[:-1], reweighting_factors=_reweighting).fetch_model()

    def test_effective_counting(self):
        dtraj = np.array([0, 0, 0, 0, 1, 1, 0, 1])
        _reweighting = (np.array([1, 1, 1, 1, 1, 1, 1, 1]),np.array([1, 1, 1, 1, 1, 1, 1, 1]))
        estimator = GirsanovReweightingEstimator(lagtime=2, count_mode="effective")
        with self.assertRaises(ValueError):
            estimator.fit(dtraj[:-1], reweighting_factors=_reweighting).fetch_model()