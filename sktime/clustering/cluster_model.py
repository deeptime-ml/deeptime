import numpy as np
from sktime.base import Model


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
        X = np.require(data, dtype=np.float32, requirements='C')
        # TODO: consider this to be a singleton!
        if not hasattr(self, '_ext'):
            from ._ext import ClusteringBase_f
            self._ext = ClusteringBase_f(self.metric)
        dtraj = self._ext.assign(X, self.cluster_centers, n_jobs)
        res = dtraj[:, None]  # always return a column vector in this function
        return res
