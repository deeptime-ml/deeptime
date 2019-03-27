import random
import unittest

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import ParameterGrid

from sktime.clustering import KmeansClustering
from sktime.clustering.cluster_model import ClusterModel


def cluster_kmeans(data, k, max_iter=5, init_strategy='kmeans++', fixed_seed=False, n_jobs=0, cluster_centers=None,
                   callback_init_centers=None, callback_loop=None) -> (KmeansClustering, ClusterModel):
    est = KmeansClustering(n_clusters=k, max_iter=max_iter, init_strategy=init_strategy,
                           fixed_seed=fixed_seed, n_jobs=n_jobs, initial_centers=cluster_centers)
    est.fit(data, callback_init_centers=callback_init_centers, callback_loop=callback_loop)
    model = est.fetch_model()
    return est, model


class TestKmeans(unittest.TestCase):

    def test_3gaussian_1d_singletraj(self):
        # generate 1D data from three gaussians
        state = np.random.RandomState(42)
        X = [state.randn(200) - 2.0,
             state.randn(200),
             state.randn(200) + 2.0]
        X = np.atleast_2d(np.hstack(X)).T
        X = X.astype(np.float32)
        k = 50
        grid = ParameterGrid({'init_strategy': ['uniform', 'kmeans++'], 'fixed_seed': [463498, True]})
        for param in grid:
            init_strategy = param['init_strategy']
            fixed_seed = param['fixed_seed']
            kmeans, model = cluster_kmeans(X, k=k, init_strategy=init_strategy, n_jobs=1, fixed_seed=fixed_seed)
            cc = model.cluster_centers
            self.assertTrue(np.all(np.isfinite(cc)), "cluster centers borked for strat %s" % init_strategy)
            assert (np.any(cc < 1.0)), "failed for init_strategy=%s" % init_strategy
            assert (np.any((cc > -1.0) * (cc < 1.0))), "failed for init_strategy=%s" % init_strategy
            assert (np.any(cc > -1.0)), "failed for init_strategy=%s" % init_strategy

            km1, model1 = cluster_kmeans(X, k=k, init_strategy=init_strategy, fixed_seed=fixed_seed, n_jobs=0)
            km2, model2 = cluster_kmeans(X, k=k, init_strategy=init_strategy, fixed_seed=fixed_seed, n_jobs=0)
            self.assertEqual(len(model1.cluster_centers), k)
            self.assertEqual(len(model2.cluster_centers), k)

            # check initial centers (after kmeans++, uniform init) are equal.
            np.testing.assert_equal(km1.initial_centers, km2.initial_centers, err_msg='not eq for {} and seed={}'
                                    .format(init_strategy, fixed_seed))

            while not model1.converged:
                km1.fit(data=X, initial_centers=model1.cluster_centers)
            while not model2.converged:
                km2.fit(data=X, initial_centers=model1.cluster_centers)

            assert np.linalg.norm(model1.cluster_centers - km1.initial_centers) > 0
            np.testing.assert_array_almost_equal(model1.cluster_centers, model2.cluster_centers)
            np.testing.assert_allclose(model1.cluster_centers, model2.cluster_centers,
                                       err_msg="should yield same centers with fixed seed=%s for strategy %s, "
                                               "Initial centers=%s"
                                               % (fixed_seed, init_strategy, km2.initial_centers))

    def test_check_convergence_serial_parallel(self):
        """ check serial and parallel version of kmeans converge to the same centers.

        Artificial data set is created with 6 disjoint point blobs, to ensure the parallel and the serial version
        converge to the same result. If the blobs would overlap we can not guarantee this, because the parallel version
        can potentially converge to a closer point, which is chosen in a non-deterministic way (multiple threads).
        """
        k = 6
        max_iter = 50
        data = make_blobs(n_samples=500, random_state=45, centers=k, cluster_std=0.5, shuffle=False)[0]
        repeat = True
        it = 0
        # since this can fail in like one of 100 runs, we repeat until success.
        while repeat and it < 3:
            for strat in ('uniform', 'kmeans++'):
                seed = random.randint(0, 2 ** 32 - 1)
                cl_serial, model_serial = cluster_kmeans(data, k=k, n_jobs=0, fixed_seed=seed, max_iter=max_iter,
                                                         init_strategy=strat)
                cl_parallel, model_parallel = cluster_kmeans(data, k=k, n_jobs=2, fixed_seed=seed, max_iter=max_iter,
                                                             init_strategy=strat)
                try:
                    np.testing.assert_allclose(model_serial.cluster_centers, model_parallel.cluster_centers, atol=1e-4)
                    repeat = False
                except AssertionError:
                    repeat = True
                    it += 1

    def test_negative_seed(self):
        """ ensure negative seeds converted to something positive"""
        km, model = cluster_kmeans(np.random.random((10, 3)), k=2, fixed_seed=-1)
        self.assertGreaterEqual(km.fixed_seed, 0)

    def test_seed_too_large(self):
        km, model = cluster_kmeans(np.random.random((10, 3)), k=2, fixed_seed=2 ** 32)
        assert km.fixed_seed < 2 ** 32

    def test_3gaussian_2d_multitraj(self):
        # generate 1D data from three gaussians
        X1 = np.zeros((100, 2))
        X1[:, 0] = np.random.randn(100) - 2.0
        X2 = np.zeros((100, 2))
        X2[:, 0] = np.random.randn(100)
        X3 = np.zeros((100, 2))
        X3[:, 0] = np.random.randn(100) + 2.0
        X = [X1, X2, X3]
        kmeans, model = cluster_kmeans(np.concatenate(X), k=10)
        cc = model.cluster_centers
        assert (np.any(cc < 1.0))
        assert (np.any((cc > -1.0) * (cc < 1.0)))
        assert (np.any(cc > -1.0))

    def test_kmeans_equilibrium_state(self):
        initial_centersequilibrium = np.array([0, 0, 0])
        X = np.array([
            np.array([1, 1, 1], dtype=np.float32), np.array([1, 1, -1], dtype=np.float32),
            np.array([1, -1, -1], dtype=np.float32), np.array([-1, -1, -1], dtype=np.float32),
            np.array([-1, 1, 1], dtype=np.float32), np.array([-1, -1, 1], dtype=np.float32),
            np.array([-1, 1, -1], dtype=np.float32), np.array([1, -1, 1], dtype=np.float32)
        ])
        kmeans, model = cluster_kmeans(X, k=1)
        self.assertEqual(1, len(model.cluster_centers), 'If k=1, there should be only one output center.')
        msg = 'Type=' + str(type(kmeans)) + '. ' + \
              'In an equilibrium state the resulting centers should not be different from the initial centers.'
        np.testing.assert_equal(initial_centersequilibrium.squeeze(), model.cluster_centers.squeeze(), err_msg=msg)

    def test_kmeans_converge_outlier_to_equilibrium_state(self):
        initial_centersequilibrium = np.array([[2, 0, 0], [-2, 0, 0]])
        X = np.array([
            np.array([1, 1.5, 1], dtype=np.float32), np.array([1, 1, -1], dtype=np.float32),
            np.array([1, -1, -1], dtype=np.float32), np.array([-1, -1, -1], dtype=np.float32),
            np.array([-1, 1, 1], dtype=np.float32), np.array([-1, -1, 1], dtype=np.float32),
            np.array([-1, 1, -1], dtype=np.float32), np.array([1, -1, 1], dtype=np.float32)
        ])
        X = np.atleast_2d(X)
        kmeans, model = cluster_kmeans(X, k=2, cluster_centers=initial_centersequilibrium, max_iter=500, n_jobs=0)

        cl = model.cluster_centers
        assert np.all(np.abs(cl) <= 1), f"Got clustercenters {cl}"

    def test_kmeans_convex_hull(self):
        points = [
            [-212129 / 100000, -20411 / 50000, 2887 / 5000],
            [-212129 / 100000, 40827 / 100000, -5773 / 10000],
            [-141419 / 100000, -5103 / 3125, 2887 / 5000],
            [-141419 / 100000, 1 / 50000, -433 / 250],
            [-70709 / 50000, 3 / 100000, 17321 / 10000],
            [-70709 / 50000, 163301 / 100000, -5773 / 10000],
            [-70709 / 100000, -204121 / 100000, -5773 / 10000],
            [-70709 / 100000, -15309 / 12500, -433 / 250],
            [-17677 / 25000, -122471 / 100000, 17321 / 10000],
            [-70707 / 100000, 122477 / 100000, 17321 / 10000],
            [-70707 / 100000, 102063 / 50000, 2887 / 5000],
            [-17677 / 25000, 30619 / 25000, -433 / 250],
            [8839 / 12500, -15309 / 12500, -433 / 250],
            [35357 / 50000, 102063 / 50000, 2887 / 5000],
            [8839 / 12500, -204121 / 100000, -5773 / 10000],
            [70713 / 100000, -122471 / 100000, 17321 / 10000],
            [70713 / 100000, 30619 / 25000, -433 / 250],
            [35357 / 50000, 122477 / 100000, 17321 / 10000],
            [106067 / 50000, -20411 / 50000, 2887 / 5000],
            [141423 / 100000, -5103 / 3125, 2887 / 5000],
            [141423 / 100000, 1 / 50000, -433 / 250],
            [8839 / 6250, 3 / 100000, 17321 / 10000],
            [8839 / 6250, 163301 / 100000, -5773 / 10000],
            [106067 / 50000, 40827 / 100000, -5773 / 10000],
        ]
        kmeans, model = cluster_kmeans(np.asarray(points, dtype=np.float32), k=1)
        res = model.cluster_centers
        # Check hyperplane inequalities. If they are all fulfilled, the center lies within the convex hull.
        self.assertGreaterEqual(np.inner(np.array([-11785060650000, -6804069750000, -4811167325000], dtype=float),
                                         res) + 25000531219381, 0)
        self.assertGreaterEqual(
            np.inner(np.array([-1767759097500, 1020624896250, 721685304875], dtype=float), res) + 3749956484003, 0)
        self.assertGreaterEqual(np.inner(np.array([-70710363900000, -40824418500000, 57734973820000], dtype=float),
                                         res) + 199998509082907, 0)
        self.assertGreaterEqual(np.inner(np.array([70710363900000, 40824418500000, -57734973820000], dtype=float),
                                         res) + 199998705841169, 0)
        self.assertGreaterEqual(np.inner(np.array([70710363900000, -40824995850000, -28867412195000], dtype=float),
                                         res) + 149999651832937, 0)
        self.assertGreaterEqual(np.inner(np.array([-35355181950000, 20412497925000, -28867282787500], dtype=float),
                                         res) + 100001120662259, 0)
        self.assertGreaterEqual(
            np.inner(np.array([23570121300000, 13608139500000, 9622334650000], dtype=float), res) + 49998241292257,
            0)
        self.assertGreaterEqual(np.inner(np.array([0, 577350000, -204125000], dtype=float), res) + 1060651231, 0)
        self.assertGreaterEqual(np.inner(np.array([35355181950000, -20412497925000, 28867282787500], dtype=float),
                                         res) + 99997486799779, 0)
        self.assertGreaterEqual(np.inner(np.array([0, 72168750, 51030625], dtype=float), res) + 176771554, 0)
        self.assertGreaterEqual(np.inner(np.array([0, -288675000, 102062500], dtype=float), res) + 530329843, 0)
        self.assertGreaterEqual(np.inner(np.array([0, 0, 250], dtype=float), res) + 433, 0)
        self.assertGreaterEqual(np.inner(np.array([0, -144337500, -102061250], dtype=float), res) + 353560531, 0)
        self.assertGreaterEqual(np.inner(np.array([0, 0, -10000], dtype=float), res) + 17321, 0)

    def test_with_callbacks(self):
        init = 0
        iter = 0

        def callback_init():
            nonlocal init
            init += 1

        def callback_loop():
            nonlocal iter
            iter += 1

        cluster_kmeans(np.random.rand(100, 3), k=3, max_iter=2, callback_init_centers=callback_init,
                       callback_loop=callback_loop)
        assert init == 3
        assert iter == 2


class TestKmeansResume(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        state = np.random.RandomState(32)
        # three gaussians
        X = [state.randn(1000) - 2.0,
             state.randn(1000),
             state.randn(1000) + 2.0]
        cls.X = np.atleast_2d(np.hstack(X)).T

    def test_resume(self):
        """ check that we can continue with the iteration by passing centers"""
        initial_centers = np.array([[20, 42, -29]]).T
        cl, model = cluster_kmeans(self.X, cluster_centers=initial_centers,
                                   max_iter=1, k=3)

        resume_centers = model.cluster_centers
        cl.fit(self.X, initial_centers=resume_centers, max_iter=50)
        new_centers = model.cluster_centers

        true = np.array([-2, 0, 2])
        d0 = true - resume_centers
        d1 = true - new_centers

        diff = np.linalg.norm(d0)
        diff_next = np.linalg.norm(d1)

        self.assertLess(diff_next, diff, 'resume_centers=%s, new_centers=%s' % (resume_centers, new_centers))

    def test_syntetic_trivial(self):
        test_data = np.zeros((40000, 4))
        test_data[0:10000, :] = 30.0
        test_data[10000:20000, :] = 60.0
        test_data[20000:30000, :] = 90.0
        test_data[30000:, :] = 120.0

        expected = np.array([30.0] * 4), np.array([60.] * 4), np.array([90.] * 4), np.array([120.] * 4)
        cl, model = cluster_kmeans(test_data, k=4)
        found = [False] * 4
        for center in model.cluster_centers:
            for i, e in enumerate(expected):
                if np.all(center == e):
                    found[i] = True

        assert np.all(found)


if __name__ == "__main__":
    unittest.main()
