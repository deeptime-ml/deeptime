import numpy as np
from sktime.base import Model

from sktime.clustering._clustering_bindings import assign as _assign


class ClusterModel(Model):

    def __init__(self, n_clusters=0, cluster_centers=None, metric=None):
        self._n_clusters = n_clusters
        self._cluster_centers = cluster_centers
        self._metric = metric

    @property
    def cluster_centers(self):
        return self._cluster_centers

    @cluster_centers.setter
    def cluster_centers(self, value):
        self._cluster_centers = value

    @property
    def n_clusters(self):
        return self._n_clusters

    @n_clusters.setter
    def n_clusters(self, value: int):
        self._n_clusters = value

    @property
    def metric(self):
        return self._metric

    def transform(self, data, n_jobs=None):
        """get closest index of point in :attr:`cluster_centers` to x."""
        assert data.dtype == self.cluster_centers.dtype

        if n_jobs is None:
            n_jobs = 0
        dtraj = _assign(data, self.cluster_centers, n_jobs, self.metric)
        res = dtraj[:, None]  # always return a column vector in this function
        return res
