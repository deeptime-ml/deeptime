import warnings

import numpy as np

from . import metrics
from ._clustering_bindings import regspace as _regspace_ext
from .cluster_model import ClusterModel
from ..base import Estimator

__all__ = ['RegularSpace']

from ..util.parallel import handle_n_jobs


class RegularSpace(Estimator):
    """Clusters data objects in such a way, that cluster centers are at least in distance of dmin to each other
    according to the given metric. The assignment of data objects to cluster centers is performed by Voronoi partioning.

    Regular space clustering :cite:`regspace-prinz2011markov` is very similar to Hartigan's leader
    algorithm :cite:`regspace-hartigan1975clustering`. It consists
    of two passes through the data. Initially, the first data point is added to the list of centers. For every
    subsequent data point, if it has a greater distance than dmin from every center, it also becomes a center.
    In the second pass, a Voronoi discretization with the computed centers is used to partition the data.

    References
    ----------
    .. bibliography:: /references.bib
        :style: unsrt
        :filter: docname in docnames
        :keyprefix: regspace-
    """

    def __init__(self, dmin: float, max_centers: int = 1000, metric: str = 'euclidean', n_jobs=None):
        r"""
        Initializes a new regular space clustering estimator.

        Parameters
        ----------
        dmin : float
            Minimum distance between all clusters, must be non-negative.
        max_centers : int
            If this threshold is met during finding the centers, the algorithm will terminate. Must be positive.
        metric : str, default='euclidean'
            The metric to use during clustering. For a list of available metrics,
            see the :data:`metric registry <deeptime.clustering.metrics>`.
        n_jobs : int, optional, default=None
            Number of threads to use during estimation.
        """
        super(RegularSpace, self).__init__()
        self.dmin = dmin
        self.metric = metric
        self.max_centers = max_centers
        self.n_jobs = n_jobs

    @property
    def metric(self) -> str:
        r"""
        The metric that is used for clustering.

        :type: str.
        """
        return self._metric

    @metric.setter
    def metric(self, value: str):
        if value not in metrics.available:
            raise ValueError(f"Unknown metric {value}, available metrics: {metrics.available}")
        self._metric = value

    @property
    def dmin(self) -> float:
        r"""
        Minimum distance between cluster centers.

        :getter: Yields the currently set minimum distance.
        :setter: Sets a new minimum distance, must be non-negative.
        :type: float
        """
        return self._dmin

    @dmin.setter
    def dmin(self, d: float):
        if d < 0:
            raise ValueError("d has to be non-negative")

        self._dmin = d

    @property
    def max_centers(self) -> int:
        r"""
        Cutoff during clustering. If reached no more data is taken into account. You might then consider a larger
        value or a larger dmin value.

        :getter: Current maximum number of cluster centers.
        :setter: Sets a new maximum number of cluster centers, must be non-negative.
        :type: int
        """
        return self._max_centers

    @max_centers.setter
    def max_centers(self, value: int):
        if value <= 0:
            raise ValueError("max_centers has to be positive")

        self._max_centers = value

    @property
    def n_clusters(self) -> int:
        r""" Alias to :attr:`max_centers`. """
        return self.max_centers

    @n_clusters.setter
    def n_clusters(self, val: int):
        self.max_centers = val

    @property
    def n_jobs(self) -> int:
        r"""
        The number of threads to use during estimation.

        :getter: Yields the number of threads to use, -1 is an allowed value for all available threads.
        :setter: Sets the number of threads to use, can be None in which case it defaults to 1 thread.
        :type: int
        """
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, value):
        self._n_jobs = handle_n_jobs(value)

    def fetch_model(self) -> ClusterModel:
        """
        Fetches the current model. Can be `None` in case :meth:`fit` was not called yet.

        Returns
        -------
        model : ClusterModel or None
            The latest estimated model or None.
        """
        return self._model

    def fit(self, data, n_jobs=None):
        r"""
        Fits this estimator onto data. The estimation is carried out by

        #. Choosing first data frame as centroid
        #. for all frames :math:`x\in X`: Calculate distance to all cluster centers
        #. Add a new centroid if minimal distance to all other cluster centers is larger or equal :attr:`dmin`.

        Parameters
        ----------
        data : (T, n) ndarray
            the data to fit
        n_jobs : int, optional, default=None
            Number of jobs, superseeds :attr:`n_jobs` if set to an integer value

        Returns
        -------
        self : RegularSpace
            reference to self
        """

        n_jobs = self.n_jobs if n_jobs is None else n_jobs
        if n_jobs is None:
            n_jobs = 1

        if data.ndim == 1:
            data = data[:, np.newaxis]

        clustercenters = []
        converged = False
        try:
            _regspace_ext.cluster(data, clustercenters, self.dmin, self.max_centers, n_jobs, metrics[self.metric]())
            converged = True
        except _regspace_ext.MaxCentersReachedException:
            warnings.warn('Maximum number of cluster centers reached.'
                          ' Consider increasing max_centers or choose'
                          ' a larger minimum distance, dmin.')
        finally:
            # even if not converged, we store the found centers.
            clustercenters = np.asarray_chkfinite(clustercenters).squeeze()

            self._model = ClusterModel(len(clustercenters), clustercenters, self.metric, converged)

            if len(clustercenters) == 1:
                warnings.warn('Have found only one center according to '
                              'minimum distance requirement of %f' % self.dmin)

        return self
