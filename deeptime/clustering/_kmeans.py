import random
import warnings
from typing import Optional

import numpy as np

from ..base import EstimatorTransformer
from ._cluster_model import ClusterModel
from . import _clustering_bindings as _bd, metrics

from ..util.parallel import handle_n_jobs


def kmeans_plusplus(data, n_clusters: int, metric: str = 'euclidean', callback=None, seed: int = -1,
                    n_jobs: Optional[int] = None):
    r""" Performs kmeans++ initialization. :footcite:`arthur2006k`

    Parameters
    ----------
    data : np.ndarray
        Input data of shape (T, n_dim).
    n_clusters : int
        The number of cluster centers.
    metric : str, default='euclidean'
        Metric to use during clustering, default evaluates to euclidean metric. For a list of available metrics,
        see the :data:`metric registry <deeptime.clustering.metrics>`.
    callback: callable or None
        used for kmeans++ initialization to indicate progress, called once per assigned center.
    seed : int, optional, default=-1
        The random seed. If non-negative, this fixes the random generator's seed and makes results reproducible.
    n_jobs : int, optional, default=None
        Number of jobs.

    Returns
    -------
    centers : np.ndarray
        An (n_centers, dim)-shaped array with a kmeans++ cluster center initial guess.

    References
    ----------
    .. footbibliography::
    """
    n_jobs = handle_n_jobs(n_jobs)
    metric = metrics[metric]()
    return _bd.kmeans.init_centers_kmpp(data, k=n_clusters, random_seed=seed, n_threads=n_jobs,
                                        callback=callback, metric=metric)


class KMeansModel(ClusterModel):
    r"""The K-means clustering model. Stores all important information which are result of the estimation procedure.
    It can also be used to transform data by assigning each frame to its closest cluster center. For an example
    please see the documentation of the superclass :class:`ClusterModel`.

    Parameters
    ----------
    cluster_centers : (k, d) ndarray
        The d-dimensional cluster centers, length of the array should coincide with :attr:`n_clusters`.
    metric : str
        The metric that was used
    tolerance : float, optional, default=None
        Tolerance which was used as convergence criterium. Defaults to `None` so that clustering models
        can be constructed purely from cluster centers and metric.
    inertias : (t,) ndarray or None, optional, default=None
        Value of the cost function over :code:`t` iterations. Defaults to `None` so that clustering models
        can be constructed purely from cluster centers and metric.
    converged : bool, optional, default=False
        Whether the convergence criterium was met.

    See Also
    --------
    ClusterModel
    KMeans
    MiniBatchKMeans
    """

    def __init__(self, cluster_centers, metric: str, tolerance: Optional[float] = None,
                 inertias: Optional[np.ndarray] = None, converged: bool = False):
        super().__init__(cluster_centers, metric, converged=converged)
        self._inertias = inertias
        self._tolerance = tolerance

    @property
    def tolerance(self):
        """
        The tolerance used as stopping criterion in the kmeans clustering loop. In particular, when
        the relative change in the inertia is smaller than the given tolerance value.

        Returns
        -------
        tolerance : float
            the tolerance
        """
        return self._tolerance

    @property
    def inertia(self) -> Optional[int]:
        r"""Sum of squared distances to assigned centers of training data

        .. math:: \sum_{i=1}^k \sum_{x\in S_i} d(x, \mu_i)^2,

        where :math:`x` are the frames assigned to their respective cluster center :math:`S_i`.

        :type: float or None
        """
        if self._inertias is not None and len(self._inertias) > 0:
            return self._inertias[-1]
        else:
            return None

    @property
    def inertias(self) -> Optional[np.ndarray]:
        r""" Series of inertias over the the iterations of k-means.

        :type: (t, dtype=float) ndarray or None
        """
        return self._inertias

    def score(self, data: np.ndarray, n_jobs: Optional[int] = None) -> float:
        r""" Computes how well the model fits to given data by computing the
        :meth:`inertia <deeptime.clustering.kmeans.KMeansModel.inertia>`.

        Parameters
        ----------
        data : (T, d) ndarray, dtype=float or double
            dataset with T entries and d dimensions
        n_jobs : int, optional, default=None
            number of jobs to use

        Returns
        -------
        score : float
            the inertia
        """
        n_jobs = handle_n_jobs(n_jobs)
        return _bd.kmeans.cost_function(data, self.cluster_centers, n_jobs, metrics[self.metric]())


class KMeans(EstimatorTransformer):
    r"""Clusters the data in a way that minimizes the cost function

    .. math:: C(S) = \sum_{i=1}^{k} \sum_{\mathbf{x}_j \in S_i} \left\| \mathbf{x}_j - \boldsymbol\mu_i \right\|^2

    where :math:`S_i` are clusters with centers of mass :math:`\mu_i` and :math:`\mathbf{x}_j` data points
    associated to their clusters.

    The outcome is very dependent on the initialization, in particular we offer "kmeans++" and "uniform". The latter
    picks initial centers random-uniformly over the provided data set. The former tries to find an initialization
    which is covering the spatial configuration of the dataset more or less uniformly. For details
    see :footcite:`arthur2006k`.

    Parameters
    ----------
    n_clusters : int
        amount of cluster centers.
    max_iter : int, default=500
        maximum number of iterations before stopping.
    metric : str, default='euclidean'
        Metric to use during clustering, default evaluates to euclidean metric. For a list of available metrics,
        see the :data:`metric registry <deeptime.clustering.metrics>`.
    tolerance : float, default=1e-5
        Stop iteration when the relative change in the cost function (inertia)

        .. math:: C(S) = \sum_{i=1}^{k} \sum_{\mathbf x \in S_i} \left\| \mathbf x - \boldsymbol\mu_i \right\|^2

        is smaller than tolerance.
    init_strategy : str, default='kmeans++'
        one of 'kmeans++', 'uniform'; determining how the initial cluster centers are being chosen
    fixed_seed : bool or int, default=False
        if True, the seed gets set to 42. Use time based seeding otherwise. If an integer is given, use this to
        initialize the random generator.
    n_jobs : int or None, default=None
        Number of threads to use during clustering and assignment of data. If None, one core will be used.
    initial_centers: None or np.ndarray[k, dim], default=None
        This is used to resume the kmeans iteration. Note, that if this is set, the init_strategy is ignored and
        the centers are directly passed to the kmeans iteration algorithm.

    References
    ----------
    .. footbibliography::

    See Also
    --------
    KMeansModel
    MiniBatchKMeans
    """

    def __init__(self, n_clusters: int, max_iter: int = 500, metric='euclidean',
                 tolerance=1e-5, init_strategy: str = 'kmeans++', fixed_seed=False,
                 n_jobs=None, initial_centers=None):
        super(KMeans, self).__init__()

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.metric = metric
        self.tolerance = tolerance
        self.init_strategy = init_strategy
        self.fixed_seed = fixed_seed
        self.random_state = np.random.RandomState(self.fixed_seed)
        self.n_jobs = handle_n_jobs(n_jobs)
        self.initial_centers = initial_centers

    @property
    def initial_centers(self) -> Optional[np.ndarray]:
        r"""
        Yields initial centers which override the :meth:`init_strategy`. Can be used to resume k-means iterations.

        :getter: The initial centers or None.
        :setter: Sets the initial centers. If not None, the array is expected to have length :attr:`n_clusters`.
        :type: (k, n) ndarray or None
        """
        return self._initial_centers

    @initial_centers.setter
    def initial_centers(self, value: Optional[np.ndarray]):
        if value is not None and value.shape[0] != self.n_clusters:
            raise ValueError("initial centers must be None or of shape (k, d) where k is the number of cluster centers."
                             " Expected k={}, got {}.".format(self.n_clusters, value.shape[0]))
        self._initial_centers = value

    @property
    def n_jobs(self) -> int:
        r"""
        Number of threads to use during clustering and assignment of data.

        :getter: Yields the number of threads. If -1, all available threads are used.
        :setter: Sets the number of threads to use. If -1, use all, if None, use 1.
        :type: int
        """
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, value: Optional[int]):
        self._n_jobs = handle_n_jobs(value)

    @property
    def n_clusters(self) -> int:
        r"""
        The number of cluster centers to use.

        :getter: Yields the number of cluster centers.
        :setter: Sets the number of cluster centers.
        :type: int
        """
        return self._n_clusters

    @n_clusters.setter
    def n_clusters(self, value: int):
        self._n_clusters = value

    @property
    def max_iter(self) -> int:
        r"""
        Maximum number of clustering iterations before stop.

        :getter: Yields the maximum number of clustering iterations
        :setter: Sets the max. number of clustering iterations
        :type: int
        """
        return self._max_iter

    @max_iter.setter
    def max_iter(self, value: int):
        self._max_iter = value

    @property
    def tolerance(self) -> float:
        r"""
        Stopping criterion for the k-means iteration. When the relative change of the cost function between two
        iterations is less than the tolerance, the algorithm is considered to be converged.

        :getter: Yields the currently set tolerance.
        :setter: Sets a new tolerance.
        :type: float
        """
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value: float):
        self._tolerance = value

    @property
    def metric(self) -> str:
        r"""
        The metric that is used for clustering.

        See Also
        --------
        _clustering_bindings.Metric : The metric class, can be subclassed
        metrics : Metrics registry which maps from metric label to actual implementation
        """
        return self._metric

    @metric.setter
    def metric(self, value: str):
        if value not in metrics.available:
            raise ValueError(f"Unknown metric {value}, available metrics: {metrics.available}")
        self._metric = value

    def fetch_model(self) -> Optional[KMeansModel]:
        """
        Fetches the current model. Can be `None` in case :meth:`fit` was not called yet.

        Returns
        -------
        model : KMeansModel or None
            the latest estimated model
        """
        return self._model

    def transform(self, data, **kw) -> np.ndarray:
        """
        Transforms a trajectory to a discrete trajectory by assigning each frame to its respective cluster center.

        Parameters
        ----------
        data : (T, n) ndarray
            trajectory with `T` frames and data points in `n` dimensions.
        **kw
            ignored kwargs for scikit-learn compatibility

        Returns
        -------
        discrete_trajectory : (T, 1) ndarray
            discrete trajectory

        See Also
        --------
        ClusterModel.transform : transform method of cluster model, implicitly called.
        """
        return super().transform(data)

    @property
    def init_strategy(self):
        r"""Strategy to get an initial guess for the centers.

        :getter: Yields the strategy, can be one of "kmeans++" or "uniform".
        :setter: Setter for the initialization strategy that is used when no initial centers are provided.
        :type: string
        """
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

        Fix this to get reproducible results in conjunction with n_jobs=0. The latter is needed, because parallel
        execution causes non-deterministic behaviour again.
        """
        return self._fixed_seed

    @fixed_seed.setter
    def fixed_seed(self, value: [bool, int, None]):
        """
        Sets a fixed seed for cluster estimation to get reproducible results. This only works in the case of n_jobs=0
        or n_jobs=1, parallel execution reintroduces non-deterministic behavior.

        Parameters
        ----------
        value : bool or int or None
            If the value is `True`, the seed will be fixed on `42`, if it is `False` or `None`, the seed gets drawn
            randomly. In case an `int` value is provided, that will be used as fixed seed.

        """
        if isinstance(value, bool) or value is None:
            if value:
                self._fixed_seed = 42
            else:
                self._fixed_seed = random.randint(0, 2 ** 32 - 1)
        elif isinstance(value, int):
            if value < 0 or value > 2 ** 32 - 1:
                warnings.warn("seed has to be non-negative (or smaller than 2**32)."
                              " Seed will be chosen randomly.")
                self.fixed_seed = False
            else:
                self._fixed_seed = value
        else:
            raise ValueError("fixed seed has to be None, bool or integer")

    def _pick_initial_centers(self, data, strategy, n_jobs, callback=None):
        if self.n_clusters > len(data):
            raise ValueError('Not enough data points for desired amount of clusters.')

        if strategy == 'uniform':
            return data[self.random_state.randint(0, len(data), size=self.n_clusters)]
        elif self.init_strategy == 'kmeans++':
            return kmeans_plusplus(data, self.n_clusters, self.metric,
                                   callback=callback, seed=self.fixed_seed, n_jobs=n_jobs)

    def fit(self, data, initial_centers=None, callback_init_centers=None, callback_loop=None, n_jobs=None):
        """ Perform the clustering.

        Parameters
        ----------
        data: np.ndarray
            data to be clustered, shape should be (N, D), where N is the number of data points, D the dimension.
            In case of one-dimensional data, a shape of (N,) also works.
        initial_centers: np.ndarray or None
            Optional cluster center initialization that supersedes the estimator's `initial_centers` attribute
        callback_init_centers: function or None
            used for kmeans++ initialization to indicate progress, called once per assigned center.
        callback_loop: function or None
            used to indicate progress on kmeans iterations, called once per iteration.
        n_jobs: None or int
            if not None, supersedes the n_jobs attribute of the estimator instance; must be non-negative

        Returns
        -------
        self : KMeans
            reference to self
        """
        if data.ndim == 1:
            data = data[:, np.newaxis]
        n_jobs = handle_n_jobs(self.n_jobs if n_jobs is None else n_jobs)
        if initial_centers is not None:
            self.initial_centers = initial_centers
        if self.initial_centers is None:
            self.initial_centers = self._pick_initial_centers(data, self.init_strategy, n_jobs, callback_init_centers)

        # run k-means with all the data
        converged = False
        cluster_centers, code, iterations, cost = _bd.kmeans.cluster_loop(
            data, self.initial_centers.copy(), n_jobs, self.max_iter,
            self.tolerance, callback_loop, metrics[self.metric]())
        if code == 0:
            converged = True
        else:
            warnings.warn(f"Algorithm did not reach convergence criterion"
                          f" of {self.tolerance} in {self.max_iter} iterations. Consider increasing max_iter.")
        self._model = KMeansModel(metric=self.metric, tolerance=self.tolerance,
                                  cluster_centers=cluster_centers, inertias=cost, converged=converged)

        return self


class MiniBatchKMeans(KMeans):
    r""" K-means clustering in a mini-batched fashion.

    Parameters
    ----------
    batch_size : int, optional, default=100
        The maximum sample size if calling :meth:`fit()`.

    See Also
    --------
    KMeans : Superclass, see for description of remaining parameters.
    KMeansModel
    """

    def __init__(self, n_clusters, batch_size=100, max_iter=5, metric='euclidean', tolerance=1e-5,
                 init_strategy='kmeans++', n_jobs=None, initial_centers=None):
        super(MiniBatchKMeans, self).__init__(n_clusters, max_iter, metric,
                                              tolerance, init_strategy, False,
                                              n_jobs=n_jobs,
                                              initial_centers=initial_centers)
        self.batch_size = batch_size

    def fit(self, data, initial_centers=None, callback_init_centers=None, callback_loop=None, n_jobs=None):
        r""" Perform clustering on whole data. """
        n_jobs = handle_n_jobs(self.n_jobs if n_jobs is None else n_jobs)
        if data.ndim == 1:
            data = data[:, np.newaxis]
        if initial_centers is not None:
            self.initial_centers = initial_centers
        if self.initial_centers is None:
            self.initial_centers = self._pick_initial_centers(data, self.init_strategy, n_jobs, callback_init_centers)
        indices = np.arange(len(data))
        self._model = KMeansModel(cluster_centers=self.initial_centers, metric=self.metric,
                                  tolerance=self.tolerance, inertias=np.array([float('inf')]))
        for epoch in range(self.max_iter):
            np.random.shuffle(indices)
            if len(data) > self.batch_size:
                split = np.array_split(indices, len(data) // self.batch_size)
            else:
                split = [indices]
            for chunk_indices in split:
                self.partial_fit(data[chunk_indices], n_jobs=n_jobs)
                if self._model.converged:
                    break
        return self

    def partial_fit(self, data, n_jobs=None):
        r"""
        Updates the current model (or creates a new one) with data. This method can be called repeatedly and thus
        be used to train a model in an on-line fashion. Note that usually multiple passes over the data is used.
        Also this method should not be mixed with calls to :meth:`fit`, as then the model is overwritten with a new
        instance based on the data passed to :meth:`fit`.

        Parameters
        ----------
        data : (T, n) ndarray
            Data with which the model is updated and/or initialized.
        n_jobs : int, optional, default=None
            number of jobs to use when updating the model, supersedes the n_jobs attribute of the estimator.

        Returns
        -------
        self : MiniBatchKMeans
            reference to self
        """
        n_jobs = handle_n_jobs(self.n_jobs if n_jobs is None else n_jobs)
        if self.initial_centers is None:
            # we have no initial centers set, pick some based on the first partial fit
            self.initial_centers = self._pick_initial_centers(data, self.init_strategy, n_jobs)

        if self._model is None:
            self._model = KMeansModel(cluster_centers=np.copy(self.initial_centers), metric=self.metric,
                                      tolerance=self.tolerance, inertias=np.array([float('inf')]))
        if data.ndim == 1:
            data = data[:, np.newaxis]
        metric_instance = metrics[self.metric]()
        self._model._cluster_centers = _bd.kmeans.cluster(data, self._model.cluster_centers, n_jobs, metric_instance)[0]
        cost = _bd.kmeans.cost_function(data, self._model.cluster_centers, n_jobs, metric_instance)

        rel_change = np.abs(cost - self._model.inertia) / cost if cost != 0.0 else 0.0
        self._model._inertias = np.append(self._model._inertias, cost)

        if rel_change <= self.tolerance:
            self._model._converged = True

        return self
