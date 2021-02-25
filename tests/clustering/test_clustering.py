import numpy as np
import pytest
from numpy.testing import assert_equal

from deeptime.clustering import ClusterModel, KMeans, RegularSpace, MiniBatchKMeans


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


@pytest.mark.parametrize("dim", [1, 2, 5], ids=lambda x: f"dim={x}")
@pytest.mark.parametrize("algo", ["kmeans", "regspace", "minibatch-kmeans"])
def test_one_cluster_edgecase(algo, dim):
    data = np.random.uniform(size=(2, dim))
    clustering = None
    if algo == 'kmeans':
        clustering = KMeans(n_clusters=1).fit(data).fetch_model()
    elif algo == 'regspace':
        clustering = RegularSpace(dmin=10, max_centers=1).fit(data).fetch_model()
    elif algo == 'minibatch-kmeans':
        clustering = MiniBatchKMeans(n_clusters=1).fit(data).fetch_model()
    else:
        pytest.fail()
    assert clustering is not None
    assert_equal(clustering.n_clusters, 1)
    assert_equal(clustering.dim, dim)
    dtraj = clustering.transform(data)
    assert_equal(len(dtraj), 2)
    assert_equal(dtraj[0], 0)
    assert_equal(dtraj[1], 0)
