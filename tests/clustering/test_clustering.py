import numpy as np
import pytest
from numpy.testing import assert_equal

from deeptime.clustering import ClusterModel


@pytest.mark.parametrize("ndim", [0, 1, 2], ids=lambda x: f"ndim={x}")
@pytest.mark.parametrize("njobs", [None, 1, 2], ids=lambda x: f"njobs={x}")
def test_ndim_assignment(ndim, njobs):
    centers = np.random.uniform(size=(15, ndim)).squeeze()
    model = ClusterModel(centers)
    assert_equal(model.dim, ndim)
    data = np.random.uniform(size=(50, ndim)).squeeze()
    dtraj = model.transform(data, n_jobs=njobs)
    if data.ndim == 1:
        data = data[..., None]
    for i in range(len(data)):
        cc = dtraj[i]
        x = data[i]
        dists = np.linalg.norm(model.cluster_centers - x[None, :], axis=1)
        assert_equal(cc, np.argmin(dists))
