import random
import warnings

import numpy as np

from sktime.base import Estimator
from sktime.clustering.cluster_model import ClusterModel
from sktime.util import get_n_jobs

__all__ = ['KmeansClustering', 'MiniBatchKmeansClustering']


class KmeansClustering(Estimator):
    r"""Kmeans clustering

    Parameters
    ----------
    n_clusters : int
        amount of cluster centers.

    max_iter : int
        maximum number of iterations before stopping.

    tolerance : float
        stop iteration when the relative change in the cost function

        .. math:: C(S) = \sum_{i=1}^{k} \sum_{\mathbf x \in S_i} \left\| \mathbf x - \boldsymbol\mu_i \right\|^2

        is smaller than tolerance.
    metric : str
        metric to use during clustering ('euclidean', 'minRMSD')

    init_strategy : string
        can be either 'kmeans++' or 'uniform', determining how the initial
        cluster centers are being chosen

    fixed_seed : bool or int
        if True, the seed gets set to 42. Use time based seeding otherwise.
        if an integer is given, use this to initialize the random generator.

    n_jobs : int or None, default None
        Number of threads to use during assignment of the data.
        If None, all available CPUs will be used.

    initial_centers: None or array(k, dim)
        This is used to resume the kmeans iteration. Note, that if this is set, the init_strategy is ignored and
        the centers are directly passed to the kmeans iteration algorithm.
    """
    def __init__(self, n_clusters, max_iter=5, metric='euclidean',
                 tolerance=1e-5, init_strategy='kmeans++', fixed_seed=False,
                 n_jobs=None, initial_centers=None, random_state=None):
        if initial_centers is None:
            initial_centers = ()

        if random_state is None:
            random_state = np.random.RandomState()

        if n_jobs is None:
            n_jobs = get_n_jobs()

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.metric = metric
        self.tolerance = tolerance
        self.init_strategy = init_strategy
        self.fixed_seed = fixed_seed
        self.n_jobs = n_jobs
        self.initial_centers = initial_centers
        self.random_state = random_state

        super(KmeansClustering, self).__init__()

    def _create_model(self) -> ClusterModel:
        return ClusterModel(n_clusters=self.n_clusters, metric=self.metric)

    @property
    def init_strategy(self):
        """Strategy to get an initial guess for the centers."""
        return self._init_strategy

    @init_strategy.setter
    def init_strategy(self, value: str):
        valid = ('kmeans++', 'uniform')
        if value not in valid:
            raise ValueError('invalid parameter "{}" for init_strategy. Should be one of {}'.format(value, valid))
        self._init_strategy = value

    @property
    def fixed_seed(self):
        """ seed for random choice of initial cluster centers.

        Fix this to get reproducible results in conjunction with n_jobs=1. The latter is needed, because parallel
        execution causes non-deterministic behaviour again.
        """
        return self._fixed_seed

    @fixed_seed.setter
    def fixed_seed(self, value: [bool, int, None]):
        if isinstance(value, bool) or value is None:
            if value:
                self._fixed_seed = 42
            else:
                self._fixed_seed = random.randint(0, 2 ** 32 - 1)
        elif isinstance(value, int):
            if value < 0 or value > 2 ** 32 - 1:
                warnings.warn("seed has to be positive (or smaller than 2**32-1)."
                              " Seed will be chosen randomly.")
                self.fixed_seed = False
            else:
                self._fixed_seed = value
        else:
            raise ValueError("fixed seed has to be bool or integer")

    def fit(self, data, callback_init_centers=None, callback_loop=None):
        """

        Parameters
        ----------
        data: ndarray
            data to be clustered
        callback_init_centers: function or None
            used for kmeans++ initialization to indicate progress.
        callback_loop: function or None
            used to indicate progress on kmeans iterations.

        """
        # TODO: check data
        ndim = data.shape[1]

        from ._ext import kmeans as kmeans_mod
        # TODO: based on input type of data use Kmeans_f or kmeans_d
        ext = kmeans_mod.Kmeans_f(self.n_clusters, self.metric, ndim)

        if self.initial_centers is None:
            if self.init_strategy == 'uniform':
                if self.n_clusters > len(data):
                    raise ValueError('Not enough data points for desired amount of clusters.')
                self.initial_centers = data[self.random_state.random_integers(0, len(data), size=self.n_clusters)]
            elif self.init_strategy == 'kmeans++':
                self.initial_centers = ext.init_centers_KMpp(data, self.fixed_seed, self.n_jobs)

        # run k-means with all the data
        converged = False
        cluster_centers, code, iterations = ext.cluster_loop(data, self.initial_centers,
                                                             self.n_jobs, self.max_iter, self.tolerance,
                                                             callback_loop)
        if code == 0:
            converged = True
        else:
            warnings.warn("Algorithm did not reach convergence criterion"
                          " of %g in %i iterations. Consider increasing max_iter.",
                          self.tolerance, self.max_iter)

        self._model.converged = converged
        self._model.cluster_centers = cluster_centers

        return self


class MiniBatchKmeansClustering(KmeansClustering):
    r"""Mini-batch k-means clustering"""

    def __init__(self, n_clusters, max_iter=5, metric='euclidean', tolerance=1e-5, init_strategy='kmeans++',
                 n_jobs=None, initial_centers=None):

        super(MiniBatchKmeansClustering, self).__init__(n_clusters, max_iter, metric,
                                                        tolerance, init_strategy, False,
                                                        n_jobs=n_jobs,
                                                        initial_centers=initial_centers)

        # we need to remember this state during partial_fit calls.
        self._converged = False
        self._prev_cost = float('inf')

        from ._ext import kmeans as kmeans_mod
        # TODO: based on input type of data use Kmeans_f or kmeans_d
        self._ext = kmeans_mod.Kmeans_f(self.n_clusters, self.metric)

    def partial_fit(self, data):
        if self._model.cluster_centers is None:
            self._model.cluster_centers = np.empty((self.n_clusters, data.shape[1]))
        cluster_centers = self._model.cluster_centers

        cluster_centers = self._ext.cluster(data, cluster_centers, self.n_jobs)
        cost = self._ext.cost_function(data, cluster_centers, self.n_jobs)

        rel_change = np.abs(cost - self._prev_cost) / cost if cost != 0.0 else 0.0
        self._prev_cost = cost

        if rel_change <= self.tolerance:
            self._converged = True

        return self
