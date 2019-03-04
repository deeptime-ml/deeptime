import itertools
import unittest

import numpy as np

from sktime.clustering import RegularSpaceClustering

def RandomDataSource(a=None, b=None, n_samples=1000, dim=3):
        """
        creates random values in interval [a,b]
        """
        data = np.random.random((n_samples, dim))
        if a is not None and b is not None:
            data *= (b - a)
            data += a
        return data


class TestRegSpaceClustering(unittest.TestCase):

    def setUp(self):
        self.dmin = 0.3
        self.clustering = RegularSpaceClustering(dmin=self.dmin)
        self.src = RandomDataSource()

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

        assert len(model.clustercenters) > 1

        # num states == num _clustercenters?
        self.assertEqual(len(np.unique(model.dtrajs)),  len(
            model.clustercenters), "number of unique states in dtrajs"
            " should be equal.")

        data_to_cluster = np.random.random((1000, 3))
        model.assign(data_to_cluster, stride=1)

    def test_spread_data(self):
        src = RandomDataSource(a=-2, b=2)
        self.clustering.dmin = 2
        self.clustering.fit(src)

    def test1d_data(self):
        data = np.random.random(100)
        RegularSpaceClustering(dmin=0.3).fit(data)

    def test_non_existent_metric(self):
        self.clustering.metric = "non_existent_metric"
        with self.assertRaises(ValueError):
            self.clustering.fit(self.src)

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
            assert len(model.clustercenters) == max_centers

    def test_regspace_nthreads(self):
        for metric in ('euclidean', 'minRMSD'):
            self.clustering.fit(self.src, n_jobs=1, dmin=self.dmin, metric=metric)
            cl2 = cluster_regspace(self.src, n_jobs=2, dmin=self.dmin, metric=metric)
            np.testing.assert_equal(self.clustering.clustercenters, cl2.clustercenters)


if __name__ == "__main__":
    unittest.main()
