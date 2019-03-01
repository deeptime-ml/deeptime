"""
Created on 22.01.2015

@author: clonker, marscher, noe
"""

import math
import random
import warnings

import numpy as np
from pyemma.util.contexts import random_seed

from sktime.base import Estimator

__all__ = ['KmeansClustering', 'MiniBatchKmeansClustering']


class KmeansClustering(Estimator):
    r"""Kmeans clustering

    Parameters
    ----------
    n_clusters : int
        amount of cluster centers. When not specified (None), min(sqrt(N), 5000) is chosen as default value,
        where N denotes the number of data points

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
                 n_jobs=None, initial_centers=None):

        super(KmeansClustering, self).__init__()

        if initial_centers is None:
            initial_centers = []

        self._converged = False

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.init_strategy = init_strategy
        self.fixed_seed = fixed_seed
        self.initial_centers = initial_centers
        self.n_jobs = n_jobs

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
        """ seed for random choice of initial cluster centers. Fix this to get reproducible results."""
        return self._fixed_seed

    @fixed_seed.setter
    def fixed_seed(self, value: [bool, int]):
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

    # TODO: model property
    @property
    def converged(self):
        return self._converged

    def fit(self, data, callback_init_centers=None, callback_loop=None):
        """

        Parameters
        ----------
        data: list of arrays or array
        callback_init_centers: function or None
            used for kmeans++ initialization to indicate progress.
        callback_loop: function or None
            used to indicate progress on kmeans iterations.

        """
        # TODO: check data


        # or the centers are not initialized.
            # first pass: gather data and run k-means
        for X in data:
            if self.init_strategy == 'uniform':
                # needed for concatenation
                if len(self.clustercenters) == 0:
                    self.clustercenters = np.empty((0, X.shape[1]))

                if itraj in list(self._init_centers_indices.keys()):
                    for l in range(len(X)):
                        if len(self.clustercenters) < self.n_clusters and t + l in self._init_centers_indices[itraj]:
                            new = np.vstack((self.clustercenters, X[l]))
                            self.clustercenters = new
            elif self.init_strategy == 'kmeans++':
                self.clustercenters = self._inst.init_centers_KMpp(data, self.fixed_seed, self.n_jobs)
            self.initial_centers_ = self.clustercenters[:]

            if len(self.clustercenters) != self.n_clusters:
                # TODO: this can be non-fatal, because the extension can handle it?!
                raise RuntimeError('Passed clustercenters do not match n_clusters: {} vs. {}'.
                                   format(len(self.clustercenters), self.n_clusters))
        # run k-means with all the data
        clustercenters, code, iterations = self._inst.cluster_loop(data, self.clustercenters,
                                                                   self.n_jobs, self.max_iter, self.tolerance,
                                                                   callback_loop)
        if code == 0:
            converged = True
        else:
            warnings.warn("Algorithm did not reach convergence criterion"
                          " of %g in %i iterations. Consider increasing max_iter.",
                          self.tolerance, self.max_iter)

        self._model.converged = converged
        self._model.cluster_centers = clustercenters

        return self

    def _init_estimate(self):
        # mini-batch sets stride to None
        ###### init
        self._init_centers_indices = {}
        traj_lengths = self.trajectory_lengths(stride=stride, skip=self.skip)
        total_length = sum(traj_lengths)
        if self.init_strategy == 'kmeans++':
            self._progress_register(self.n_clusters,
                                    description="initialize kmeans++ centers", stage=0)
        self._progress_register(self.max_iter, description="kmeans iterations", stage=1)
        self._init_in_memory_chunks(total_length)

        if self.init_strategy == 'uniform':
            # gives random samples from each trajectory such that the cluster centers are distributed percentage-wise
            # with respect to the trajectories length
            with random_seed(self.fixed_seed):
                for idx, traj_len in enumerate(traj_lengths):
                    self._init_centers_indices[idx] = random.sample(list(range(0, traj_len)), int(
                        math.ceil((traj_len / float(total_length)) * self.n_clusters)))

        from ._ext import kmeans as kmeans_mod
        # TODO: based on input type of data use Kmeans_f or kmeans_d
        self._inst = kmeans_mod.Kmeans_f(self.n_clusters, self.metric, self.data_producer.ndim)

    def _initialize_centers(self, data):



class MiniBatchKmeansClustering(KmeansClustering):
    r"""Mini-batch k-means clustering"""

    __serialize_version = 0

    def __init__(self, n_clusters, max_iter=5, metric='euclidean', tolerance=1e-5, init_strategy='kmeans++',
                 batch_size=0.2, oom_strategy='memmap', fixed_seed=False, stride=None, n_jobs=None, skip=0,
                 initial_centers=None, keep_data=False):

        if stride is not None:
            raise ValueError("stride is a dummy value in MiniBatch Kmeans")
        if batch_size > 1:
            raise ValueError("batch_size should be less or equal to 1, but was %s" % batch_size)
        if keep_data:
            raise ValueError("keep_data is a dummy value in MiniBatch Kmeans")

        super(MiniBatchKmeansClustering, self).__init__(n_clusters, max_iter, metric,
                                                        tolerance, init_strategy, False,
                                                        oom_strategy, stride=stride, n_jobs=n_jobs, skip=skip,
                                                        initial_centers=initial_centers, keep_data=False)

        self.set_params(batch_size=batch_size)

    def _init_in_memory_chunks(self, size):
        return super(MiniBatchKmeansClustering, self)._init_in_memory_chunks(self._n_samples)

    def _draw_mini_batch_sample(self):
        offset = 0
        for idx, traj_len in enumerate(self._traj_lengths):
            n_samples_traj = self._n_samples_traj[idx]
            start = slice(offset, offset + n_samples_traj)

            self._random_access_stride[start, 0] = idx * np.ones(
                n_samples_traj, dtype=int)

            # draw 'n_samples_traj' without replacement from range(0, traj_len)
            choice = np.random.choice(traj_len, n_samples_traj, replace=False)

            self._random_access_stride[start, 1] = np.sort(choice).T
            offset += n_samples_traj

        return self._random_access_stride

    def _init_estimate(self):
        self._traj_lengths = self.trajectory_lengths(skip=self.skip)
        self._total_length = sum(self._traj_lengths)
        samples = int(math.ceil(self._total_length * self.batch_size))
        self._n_samples = 0
        self._n_samples_traj = {}
        for idx, traj_len in enumerate(self._traj_lengths):
            traj_samples = int(math.floor(traj_len / float(self._total_length) * samples))
            self._n_samples_traj[idx] = traj_samples
            self._n_samples += traj_samples

        self._random_access_stride = np.empty(shape=(self._n_samples, 2), dtype=int)
        super(MiniBatchKmeansClustering, self)._init_estimate()

    def _estimate(self, iterable, **kw):
        # mini-batch kmeans does not use stride. Enforce it.
        self.stride = None
        self._init_estimate()

        i_pass = 0
        prev_cost = 0

        ra_stride = self._draw_mini_batch_sample()
        with iterable.iterator(return_trajindex=False, stride=ra_stride, skip=self.skip) as iterator, \
                self._progress_context(), self._finish_estimate():
            while not (self._converged or i_pass + 1 > self.max_iter):
                first_chunk = True
                # draw new sample and re-use existing iterator instance.
                ra_stride = self._draw_mini_batch_sample()
                iterator.stride = ra_stride
                iterator.reset()
                for X in iter(iterator):
                    # collect data
                    self._collect_data(X, first_chunk, iterator.last_chunk)
                    # initialize cluster centers
                    if i_pass == 0 and not self._check_resume_iteration():
                        self._initialize_centers(X, iterator.current_trajindex, iterator.pos, iterator.last_chunk)
                    first_chunk = False

                # one pass over data completed
                self.clustercenters = self._inst.cluster(self._in_memory_chunks, self.clustercenters, self.n_jobs)
                cost = self._inst.cost_function(self._in_memory_chunks, self.clustercenters, self.n_jobs)

                rel_change = np.abs(cost - prev_cost) / cost if cost != 0.0 else 0.0
                prev_cost = cost

                if rel_change <= self.tolerance:
                    self._converged = True
                    self.logger.info("Cluster centers converged after %i steps.", i_pass + 1)
                    self._progress_force_finish(stage=1)
                else:
                    self._progress_update(1, stage=1)

                i_pass += 1

        if not self._converged:
            self.logger.info("Algorithm did not reach convergence criterion"
                             " of %g in %i iterations. Consider increasing max_iter.", self.tolerance, self.max_iter)
        return self
