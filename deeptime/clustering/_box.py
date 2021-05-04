from typing import Optional

import numpy as np

from ._cluster_model import ClusterModel
from ..base import Estimator


class BoxDiscretizationModel(ClusterModel):

    def __init__(self, cluster_centers: np.ndarray, v0, v1, n_boxes):
        super().__init__(cluster_centers)
        self.v0 = v0
        self.v1 = v1
        self.n_boxes = n_boxes

    def transform_onehot(self, data, n_jobs=None):
        dtraj = self.transform(data, n_jobs=n_jobs)
        traj_onehot = np.zeros((len(data), self.n_clusters))
        traj_onehot[np.arange(len(data)), dtraj] = 1.
        return traj_onehot


class BoxDiscretization(Estimator):

    def __init__(self, dim: int, n_boxes):
        super().__init__()
        if not isinstance(n_boxes, (list, tuple, np.ndarray)):
            if int(n_boxes) == n_boxes:
                n_boxes = [int(n_boxes)] * dim
        if len(n_boxes) != dim:
            raise ValueError(f"Dimension and number of boxes per dimension did not match ({len(n_boxes)} and {dim}).")
        self.dim = dim
        self.n_boxes = n_boxes

    def fit(self, data: np.ndarray, **kwargs):
        assert data.shape[1] == self.dim
        linspaces = []
        v0 = np.empty((self.dim,), dtype=data.dtype)
        v1 = np.empty((self.dim,), dtype=data.dtype)
        for d in range(self.dim):
            v0[d] = np.min(data[:, d])
            v1[d] = np.max(data[:, d])
            linspaces.append(np.linspace(v0[d], v1[d], num=self.n_boxes[d], endpoint=True))
        mesh = np.vstack(np.meshgrid(*tuple(linspaces))).reshape(self.dim, -1).T
        self._model = BoxDiscretizationModel(mesh, v0, v1, self.n_boxes)
        return self

    def fetch_model(self) -> Optional[BoxDiscretizationModel]:
        return super().fetch_model()
