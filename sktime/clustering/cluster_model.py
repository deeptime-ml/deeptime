import numpy as np
from sktime.base import Model


class ClusterModel(Model):

    @property
    def cluster_centers(self):
        return self._cluster_centers

    @property
    def n_clusters(self):
        return self._n_clusters

    @n_clusters.setter
    def n_clusters(self, value: int):
        self._n_clusters = value

    @property
    def metric(self):
        return self._metric

    def transform(self, data):
        """get closest index of point in :attr:`clustercenters` to x."""
        X = np.require(data, dtype=np.float32, requirements='C')
        if not hasattr(self, '_inst'):
            from ._ext import ClusteringBase_f
            self._inst = ClusteringBase_f(self.metric, X.shape[1])

        # for performance reasons we pre-center the cluster centers for minRMSD.
        # TODO: this should only reside in PyEMMA
        if self.metric == 'minRMSD' and not self._precentered:
            self._inst.precenter_centers(self.cluster_centers)
            self._precentered = True

        dtraj = self._inst.assign(X, self.cluster_centers, self.n_jobs)
        res = dtraj[:, None]  # always return a column vector in this function
        return res
