import itertools
import unittest

import numpy as np

from deeptime.clustering import RegularSpace


class TestRegSpaceClustering(unittest.TestCase):

    def setUp(self):
        self.dmin = 0.3
        self.clustering = RegularSpace(dmin=self.dmin)
        self.src = np.random.uniform(size=(1000, 3))

    def test_algorithm(self):
        model = self.clustering.fit(self.src).fetch_model()

        # assert distance for each centroid is at least dmin
        for c in itertools.combinations(model.cluster_centers, 2):
            if np.allclose(c[0], c[1]):  # skip equal pairs
                continue

            dist = np.linalg.norm(c[0] - c[1], 2)

            self.assertGreaterEqual(dist, self.dmin,
                                    "centroid pair\n%s\n%s\n has smaller"
                                    " distance than dmin(%f): %f"
                                    % (c[0], c[1], self.dmin, dist))

    def test_assignment(self):
        model = self.clustering.fit(self.src).fetch_model()

        assert len(model.cluster_centers) > 1
        dtraj = model.transform(self.src)

        # num states == num _clustercenters?
        self.assertEqual(len(np.unique(dtraj)), len(
            model.cluster_centers), "number of unique states in dtrajs"
                                    " should be equal.")

        data_to_cluster = np.random.random((1000, 3))
        model.transform(data_to_cluster)

    def test_spread_data(self):
        src = np.random.uniform(-2, 2, size=(1000, 3))
        self.clustering.dmin = 2
        self.clustering.fit(src)

    def test1d_data(self):
        data = np.random.random(100)
        RegularSpace(dmin=0.3).fit(data)

    def test_non_existent_metric(self):
        with self.assertRaises(ValueError):
            self.clustering.metric = "non_existent_metric"

    def test_too_small_dmin_should_warn(self):
        self.clustering.dmin = 1e-8
        max_centers = 50
        self.clustering.max_centers = max_centers
        import warnings
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger a warning.
            self.clustering.fit(self.src)
            assert w
            assert len(w) == 1
            model = self.clustering.fetch_model()
            assert len(model.cluster_centers) == max_centers

    def test_regspace_nthreads(self):
        self.clustering.fit(self.src, n_jobs=1)
        cl2 = RegularSpace(dmin=self.dmin, n_jobs=2).fit(self.src).fetch_model()
        centers1 = self.clustering.fetch_model().cluster_centers
        centers2 = cl2.cluster_centers
        np.testing.assert_equal(centers1, centers2)

    def test_properties(self):
        est = RegularSpace(dmin=1e-8, max_centers=500, metric='euclidean', n_jobs=5)
        np.testing.assert_equal(est.dmin, 1e-8)
        np.testing.assert_equal(est.max_centers, 500)
        np.testing.assert_equal(est.n_clusters, 500)
        est.n_clusters = 30
        np.testing.assert_equal(est.max_centers, 30)  # n_clusters and max_centers are aliases
        np.testing.assert_equal(est.metric, 'euclidean')
        np.testing.assert_equal(est.n_jobs, 5)

        with np.testing.assert_raises(ValueError):
            est.dmin = -.5  # negative, invalid!

        with np.testing.assert_raises(ValueError):
            est.metric = 'bogus'

        with np.testing.assert_raises(ValueError):
            est.max_centers = 0  # must be positive

