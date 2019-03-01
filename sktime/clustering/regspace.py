import warnings

import numpy as np

from sktime.base import Estimator
from sktime.clustering.cluster_model import ClusterModel

__all__ = ['RegularSpaceClustering']


class RegularSpaceClustering(Estimator):
    """Clusters data objects in such a way, that cluster centers are at least in
    distance of dmin to each other according to the given metric.
    The assignment of data objects to cluster centers is performed by
    Voronoi partioning.

    Regular space clustering [Prinz_2011]_ is very similar to Hartigan's leader
    algorithm [Hartigan_1975]_. It consists of two passes through
    the data. Initially, the first data point is added to the list of centers.
    For every subsequent data point, if it has a greater distance than dmin from
    every center, it also becomes a center. In the second pass, a Voronoi
    discretization with the computed centers is used to partition the data.


    Parameters
    ----------
    dmin : float
        minimum distance between all clusters.
    metric : str
        metric to use during clustering ('euclidean', 'minRMSD')
    max_centers : int
        if this cutoff is hit during finding the centers,
        the algorithm will abort.
    n_jobs : int or None, default None
        Number of threads to use during assignment of the data.
        If None, all available CPUs will be used.

    References
    ----------

    .. [Prinz_2011] Prinz J-H, Wu H, Sarich M, Keller B, Senne M, Held M, Chodera JD, Schuette Ch and Noe F. 2011.
        Markov models of molecular kinetics: Generation and Validation.
        J. Chem. Phys. 134, 174105.
    .. [Hartigan_1975] Hartigan J. Clustering algorithms.
        New York: Wiley; 1975.

    """

    def __init__(self, dmin, max_centers=1000, metric='euclidean', n_jobs=None):
        super(RegularSpaceClustering, self).__init__()
        self.dmin = dmin
        self.metric = metric
        self.max_centers = max_centers
        self.n_jobs = n_jobs

    def _create_model(self):
        return ClusterModel()

    @property
    def dmin(self):
        """Minimum distance between cluster centers."""
        return self._dmin

    @dmin.setter
    def dmin(self, d: float):
        if d < 0:
            raise ValueError("d has to be positive")

        self._dmin = d

    @property
    def max_centers(self):
        """
        Cutoff during clustering. If reached no more data is taken into account.
        You might then consider a larger value or a larger dmin value.
        """
        return self._max_centers

    @max_centers.setter
    def max_centers(self, value: int):
        if value < 0:
            raise ValueError("max_centers has to be positive")

        self._max_centers = value

    @property
    def n_clusters(self):
        return self.max_centers

    @n_clusters.setter
    def n_clusters(self, val: int):
        self.max_centers = val

    def fit(self, data, **kwargs):
        ########
        # Calculate clustercenters:
        # 1. choose first datapoint as centroid
        # 2. for all X: calc distances to all clustercenters
        # 3. add new centroid, if min(distance to all other clustercenters) >= dmin
        ########
        # temporary list to store cluster centers
        clustercenters = []
        used_data = 0
        from ._ext import regspace
        ndim = data.shape[1]
        n = data.shape[0]
        converged = False

        self._inst = regspace.Regspace_f(self.dmin, self.max_centers, self.metric, ndim)
        try:
            for X in data:
                used_data += len(X)
                self._inst.cluster(X.astype(np.float32, order='C', copy=False),
                                   clustercenters, self.n_jobs)
            converged = True
        except regspace.MaxCentersReachedException:
            warnings.warn('Maximum number of cluster centers reached.'
                          ' Consider increasing max_centers or choose'
                          ' a larger minimum distance, dmin.')
        finally:
            # even if not converged, we store the found centers.
            new_shape = (len(clustercenters), ndim)
            clustercenters = np.array(clustercenters).reshape(new_shape)
            self._model.cluster_centers = clustercenters
            self._model.n_clusters = len(clustercenters)
            self._model._converged = converged
            if len(clustercenters) == 1:
                warnings.warn('Have found only one center according to '
                              'minimum distance requirement of %f' % self.dmin)

        return self
