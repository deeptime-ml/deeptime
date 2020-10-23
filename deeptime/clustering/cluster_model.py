import numpy as np
from deeptime.base import Model, Transformer

from . import _clustering_bindings as _bd, metrics


class ClusterModel(Model, Transformer):
    r"""
    A generic clustering model. Stores the number of cluster centers, the position of cluster centers, the metric
    that was used to compute the cluster centers and whether the estimation converged. It can be used to transform
    data by assigning each frame to its closest cluster center.

    Example
    -------
    Let us create an artificial cluster model with three cluster centers in a three-dimensional space. The cluster
    centers are just the canonical basis vectors :math:`c_1 = (1,0,0)^\top`, :math:`c_2 = (0,1,0)^\top`, and
    :math:`c_3 = (0,0,1)^\top`.

    We can transform data with the model. The data are five frames sampled around the third cluster center.

    >>> model = ClusterModel(n_clusters=3, cluster_centers=np.eye(3))
    >>> data = np.random.normal(loc=[0, 0, 1], scale=0.01, size=(5, 3))
    >>> assignments = model.transform(data)
    >>> print(assignments)
    [2 2 2 2 2]
    """

    def __init__(self, n_clusters: int, cluster_centers: np.ndarray, metric: str = 'euclidean',
                 converged: bool = False):
        r"""
        Initializes a new cluster model.

        Parameters
        ----------
        n_clusters : int
            Number of cluster centers.
        cluster_centers : (k, d) ndarray
            The cluster centers, length of the array should match :attr:`n_clusters`.
        metric : str, default='euclidean'
            The metric that was used for estimation, defaults to Euclidean metric.
        converged : bool, optional, default=False
            Whether the estimation converged.
        """
        super().__init__()
        self._n_clusters = n_clusters
        self._cluster_centers = cluster_centers
        self._metric = metric
        self._converged = converged

    @property
    def cluster_centers(self) -> np.ndarray:
        """
        Gets the cluster centers that were estimated for this model.

        Returns
        -------
        np.ndarray
            Array containing estimated cluster centers.
        """
        return self._cluster_centers

    @property
    def n_clusters(self) -> int:
        """
        The number of cluster centers.

        Returns
        -------
        int
            The number of cluster centers.
        """
        return self._n_clusters

    @property
    def metric(self) -> str:
        """
        The metric that was used.

        Returns
        -------
        metric : str
            Name of the metric that was used. The name is related to the implementation via
            the :data:`metric registry <deeptime.clustering.metrics>`.
        """
        return self._metric

    @property
    def converged(self) -> bool:
        """
        Whether the estimation process converged. Per default this is set to False, which can also indicate that
        the model was created manually and does not stem from an Estimator directly.

        Returns
        -------
        converged : bool
            Whether the clustering converged
        """
        return self._converged

    def transform(self, data, n_jobs=None):
        r"""
        For each frame in `data`, yields the index of the closest point in :attr:`cluster_centers`.

        Parameters
        ----------
        data : (T, d) ndarray
            frames
        n_jobs : int, optional, default=None
            number of jobs to use for assignment

        Returns
        -------
        discrete_trajectory : (T, 1) ndarray, dtype=int
            A discrete trajectory where each frame denotes the closest cluster center.
        """
        assert data.dtype == self.cluster_centers.dtype

        if n_jobs is None:
            n_jobs = 0
        dtraj = _bd.assign(data, self.cluster_centers, n_jobs, metrics[self.metric]())
        return dtraj
