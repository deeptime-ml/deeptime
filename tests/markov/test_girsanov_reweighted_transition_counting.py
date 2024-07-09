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
            assert_equal(estimator.n_states, None)
            assert_equal(estimator.sparse, False)        

    def test_sample_counting(self):
        dtraj = np.array([0, 0, 0, 0, 1, 1, 0, 1])
        _reweighting = (np.array([1., 1., 1., 1., 1., 1., 1., 1.]),np.array([1., 1., 1., 1., 1., 1., 1., 1.]))
        estimator = GirsanovReweightingEstimator(lagtime=2, count_mode="sample")
        with self.assertRaises(ValueError):
            estimator.fit(dtraj, reweighting_factors=_reweighting).fetch_model()

    def test_sliding_counting(self):
        dtraj = np.array([0, 0, 0, 0, 1, 1, 0, 1])
        _reweighting = (np.array([1., 1., 1., 1., 1., 1., 1., 1.]),np.array([1., 1., 1., 1., 1., 1., 1., 1.]))
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

    def test_sliding_representation(self):
        dtraj = [np.array([0, 0, 1, 0, 1, 1, 0])]
        gtraj = [np.array([1., 1., 1., 1., 1., 1., 1.])]
        Mtraj = [np.array([0., 0., 0., 0., 0., 0., 0.])]
        counts_sliding = np.array([[1, 2], [2, 1]])
        counts_sliding_4nstates = np.array([[1, 2, 0, 0], [2, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        # automatic detection of n_states, sparse=False
        estimator= TransitionCountEstimator(1,'sliding')
        counts = estimator.fit(dtraj).fetch_model()
        rwght_estimator= GirsanovReweightingEstimator(1,'sliding')
        counts_rwght = rwght_estimator.fit(dtraj,(gtraj,Mtraj)).fetch_model()
        assert_equal(counts.count_matrix,counts_rwght.count_matrix)
        assert_equal(counts_sliding,counts_rwght.count_matrix)
        # automatic detection of n_states, sparse=True
        estimator= TransitionCountEstimator(1,'sliding', sparse=True)
        counts = estimator.fit(dtraj).fetch_model()
        rwght_estimator= GirsanovReweightingEstimator(1,'sliding', sparse=True)
        counts_rwght = rwght_estimator.fit(dtraj,(gtraj,Mtraj)).fetch_model()
        assert_equal(counts.state_histogram,counts_rwght.state_histogram)
        assert_equal(counts.count_matrix.data,counts_rwght.count_matrix.data)
        assert_equal(counts.count_matrix.indices,counts_rwght.count_matrix.indices)
        assert_equal(counts.count_matrix.indptr,counts_rwght.count_matrix.indptr)
        assert_equal(counts_sliding,counts_rwght.count_matrix.toarray())
        # set n_states > count_matrix shape, sparse=False
        estimator= TransitionCountEstimator(1,'sliding',n_states=4)
        counts = estimator.fit(dtraj).fetch_model()
        rwght_estimator= GirsanovReweightingEstimator(1,'sliding',n_states=4)
        counts_rwght = rwght_estimator.fit(dtraj,(gtraj,Mtraj)).fetch_model()
        assert_equal(counts.count_matrix,counts_rwght.count_matrix)
        assert_equal(counts_sliding_4nstates,counts_rwght.count_matrix)
        # set n_states > count_matrix shape, sparse=False
        estimator= TransitionCountEstimator(1,'sliding',n_states=4, sparse=True)
        counts = estimator.fit(dtraj).fetch_model()
        counts.count_matrix.toarray()
        rwght_estimator= GirsanovReweightingEstimator(1,'sliding',n_states=4, sparse=True)
        counts_rwght = rwght_estimator.fit(dtraj,(gtraj,Mtraj)).fetch_model()
        assert_equal(counts.state_histogram,counts_rwght.state_histogram)
        assert_equal(counts.count_matrix.data,counts_rwght.count_matrix.data)
        assert_equal(counts.count_matrix.indices,counts_rwght.count_matrix.indices)
        assert_equal(counts.count_matrix.indptr,counts_rwght.count_matrix.indptr)
        assert_equal(counts_sliding_4nstates,counts_rwght.count_matrix.toarray())

    def test_sliding_effective_counting(self):
        dtraj = np.array([0, 0, 0, 0, 1, 1, 0, 1])
        _reweighting = (np.array([1., 1., 1., 1., 1., 1., 1., 1.]),np.array([1., 1., 1., 1., 1., 1., 1., 1.]))
        estimator = GirsanovReweightingEstimator(lagtime=2, count_mode="sliding-effective")
        with self.assertRaises(ValueError):
            estimator.fit(dtraj, reweighting_factors=_reweighting).fetch_model()

    def test_effective_counting(self):
        dtraj = np.array([0, 0, 0, 0, 1, 1, 0, 1])
        _reweighting = (np.array([1., 1., 1., 1., 1., 1., 1., 1.]),np.array([1., 1., 1., 1., 1., 1., 1., 1.]))
        estimator = GirsanovReweightingEstimator(lagtime=2, count_mode="effective")
        with self.assertRaises(ValueError):
            estimator.fit(dtraj, reweighting_factors=_reweighting).fetch_model()
