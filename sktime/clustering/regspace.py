import warnings

import numpy as np

from base import Model
from sktime.base import Estimator
from sktime.clustering.cluster_model import ClusterModel

from sktime.clustering._clustering_bindings import Metric, EuclideanMetric
from sktime.clustering._clustering_bindings import regspace as _regspace_ext

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

    def __init__(self, dmin, max_centers=1000, metric=None, n_jobs=None):
        super(RegularSpaceClustering, self).__init__()
        self.dmin = dmin
        if metric is None:
            metric = EuclideanMetric()
        self.metric = metric
        self.max_centers = max_centers
        self.n_jobs = n_jobs

    def _create_model(self):
        return ClusterModel()

    @property
    def metric(self):
        return self._metric

    @metric.setter
    def metric(self, value):
        if value == 'euclidean':
            value = EuclideanMetric()

        if not isinstance(value, Metric):
            raise ValueError(f"Unknown metric {value}, must be subclass of _clustering_bindings.Metric")
        self._metric = value

    @property
    def dmin(self):
        """Minimum distance between cluster centers."""
        return self._dmin

    @dmin.setter
    def dmin(self, d: float):
        if d < 0:
            raise ValueError("d has to be non-negative")

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
            raise ValueError("max_centers has to be non-negative")

        self._max_centers = value

    @property
    def n_clusters(self):
        return self.max_centers

    @n_clusters.setter
    def n_clusters(self, val: int):
        self.max_centers = val

    def fetch_model(self) -> ClusterModel:
        return self._model

    def fit(self, data, n_jobs=None):
        ########
        # Calculate clustercenters:
        # 1. choose first datapoint as centroid
        # 2. for all X: calc distances to all clustercenters
        # 3. add new centroid, if min(distance to all other clustercenters) >= dmin
        ########
        # temporary list to store cluster centers

        n_jobs = self.n_jobs if n_jobs is None else n_jobs
        if n_jobs is None:
            n_jobs = 0

        if data.ndim == 1:
            data = data[:, np.newaxis]

        clustercenters = []
        converged = False
        try:
            _regspace_ext.cluster(data, clustercenters, self.dmin, self.max_centers, n_jobs, self.metric)
            converged = True
        except _regspace_ext.MaxCentersReachedException:
            warnings.warn('Maximum number of cluster centers reached.'
                          ' Consider increasing max_centers or choose'
                          ' a larger minimum distance, dmin.')
        finally:
            # even if not converged, we store the found centers.
            # new_shape = (len(clustercenters), ndim)
            clustercenters = np.asarray_chkfinite(clustercenters).squeeze() #.reshape(new_shape)
            self._model.cluster_centers = clustercenters
            self._model.n_clusters = len(clustercenters)
            self._model._converged = converged
            if len(clustercenters) == 1:
                warnings.warn('Have found only one center according to '
                              'minimum distance requirement of %f' % self.dmin)

        return self
