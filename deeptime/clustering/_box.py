from typing import Optional

import numpy as np

from ._cluster_model import ClusterModel
from ..base import Estimator


class BoxDiscretizationModel(ClusterModel):
    r""" Model produced by :class:`BoxDiscretization`. Can be used to discretize and/or one-hot transform data.

    Parameters
    ----------
    cluster_centers : ndarray
        The cluster centers.
    v0 : ndarray
        Lower left vertex of box.
    v1 : ndarray
        Upper right vertex of box.
    n_boxes : int
        Number of boxes.
    """

    def __init__(self, cluster_centers: np.ndarray, v0, v1, n_boxes):
        super().__init__(cluster_centers)
        self.v0 = v0
        self.v1 = v1
        self.n_boxes = n_boxes

    def transform_onehot(self, data, n_jobs=None):
        r"""Transforms data into discrete states with subsequent one-hot encoding.

        Parameters
        ----------
        data : ndarray
            Input data
        n_jobs : int or None, optional, default=None
            Number of jobs.

        Returns
        -------
        one_hot : ndarray
            A (T, n_boxes) shaped array with one-hot encoded data.
        """
        dtraj = self.transform(data, n_jobs=n_jobs)
        traj_onehot = np.zeros((len(data), self.n_clusters))
        traj_onehot[np.arange(len(data)), dtraj] = 1.
        return traj_onehot


class BoxDiscretization(Estimator):
    r"""An n-dimensional box discretization of Euclidean space.

    It spans an n-dimensional grid based on linspaces along each axis which is then used as cluster centers.
    The linspaces are bounded either by the user (attributes :attr:`v0` and :attr:`v1`) or estimated from data.

    Parameters
    ----------
    dim : int
        Dimension of the box discretization.
    n_boxes : int or list of int
        Number of boxes per dimension of - if given as single integer - for all dimensions.
    v0 : array or None, optional, default=None
        Lower left vertex of the box discretization. If not given this is estimated from data.
    v1 : array or None, optional, default=None
        Upper right vertex of the box discretization. If not given this is estimated from data.
    """

    def __init__(self, dim: int, n_boxes, v0=None, v1=None):
        super().__init__()
        if not isinstance(n_boxes, (list, tuple, np.ndarray)):
            if int(n_boxes) == n_boxes:
                n_boxes = [int(n_boxes)] * dim
        if len(n_boxes) != dim:
            raise ValueError(f"Dimension and number of boxes per dimension did not match ({len(n_boxes)} and {dim}).")
        if v0 is not None and len(v0) != dim:
            raise ValueError("Length of v0 did not match dimension.")
        if v1 is not None and len(v1) != dim:
            raise ValueError("Length of v1 did not match dimension.")
        self.dim = dim
        self.n_boxes = n_boxes
        self.v0 = v0
        self.v1 = v1

    def fit(self, data: np.ndarray, **kwargs):
        assert data.shape[1] == self.dim
        if self.v0 is None or self.v1 is None:
            v0 = np.empty((self.dim,), dtype=data.dtype) if self.v0 is None else self.v0
            v1 = np.empty((self.dim,), dtype=data.dtype) if self.v1 is None else self.v1
            for d in range(self.dim):
                if self.v0 is None:
                    v0[d] = np.min(data[:, d])
                if self.v1 is None:
                    v1[d] = np.max(data[:, d])
        else:
            v0 = self.v0
            v1 = self.v1
        linspaces = [np.linspace(v0[d], v1[d], num=self.n_boxes[d], endpoint=True) for d in range(self.dim)]
        mesh = np.vstack(np.meshgrid(*tuple(linspaces))).reshape(self.dim, -1).T
        self._model = BoxDiscretizationModel(mesh, v0, v1, self.n_boxes)
        return self

    def fetch_model(self) -> Optional[BoxDiscretizationModel]:
        r""" Yields the estimated model.

        Returns
        -------
        model : BoxDiscretizationModel or None
            The model.
        """
        return super().fetch_model()
