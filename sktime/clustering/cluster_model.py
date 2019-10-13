import numpy as np
from sktime.base import Model, Transformer

from sktime.clustering._clustering_bindings import assign as _assign


class ClusterModel(Model, Transformer):

    def __init__(self, n_clusters=0, cluster_centers=None, metric=None):
        self._n_clusters = n_clusters
        self._cluster_centers = cluster_centers
        self._metric = metric
        self._converged = False

    @property
    def cluster_centers(self):
        """
        Gets the cluster centers that were estimated for this model.

        Returns
        -------
        np.ndarray
            Array containing estimated cluster centers.
        """
        return self._cluster_centers

    @property
    def n_clusters(self):
        """

        Returns
        -------
        int
            The number of cluster centers.
        """
        return self._n_clusters

    @property
    def metric(self):
        return self._metric

    @property
    def converged(self):
        return self._converged

    def transform(self, data, n_jobs=None):
        """get closest index of point in :attr:`cluster_centers` to x."""
        assert data.dtype == self.cluster_centers.dtype

        if n_jobs is None:
            n_jobs = 0
        dtraj = _assign(data, self.cluster_centers, n_jobs, self.metric)
        return dtraj
