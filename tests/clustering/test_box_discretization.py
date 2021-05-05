import pytest
from numpy.testing import assert_equal, assert_raises, assert_almost_equal
import numpy as np

from deeptime.clustering import BoxDiscretization


def test_estimator_args():
    est = BoxDiscretization(dim=5, n_boxes=2)
    assert_equal(est.dim, 5)
    assert_equal(est.n_boxes, [2, 2, 2, 2, 2])

    est = BoxDiscretization(dim=5, n_boxes=[2, 3, 4, 5, 6])
    assert_equal(est.dim, 5)
    assert_equal(est.n_boxes, [2, 3, 4, 5, 6])

    with assert_raises(ValueError):
        BoxDiscretization(dim=2, n_boxes=[1, 2, 3])

    est = BoxDiscretization(dim=2, n_boxes=2, v0=[0, 0], v1=[1, 2])
    assert_equal(est.v0, [0, 0])
    assert_equal(est.v1, [1, 2])

    with assert_raises(ValueError):
        BoxDiscretization(dim=2, n_boxes=2, v0=[1, 2, 3])

    with assert_raises(ValueError):
        BoxDiscretization(dim=2, n_boxes=2, v1=[1, 2, 3])


@pytest.mark.parametrize("dim", [1, 2, 3, 4, 5, 6, 7])
def test_discretization_1box(dim):
    est = BoxDiscretization(dim=dim, n_boxes=1)
    data = np.random.uniform(size=(20, dim))
    model = est.fit(data).fetch_model()
    assert_equal(model.transform(data), [0] * len(data))
    for d in range(dim):
        assert_almost_equal(model.v0[d], np.min(data[:, d]))
        assert_almost_equal(model.v1[d], np.max(data[:, d]))


@pytest.mark.parametrize("v0_given", [False, True])
@pytest.mark.parametrize("v1_given", [False, True])
@pytest.mark.parametrize("dim", [1, 2, 3, 4, 5, 6, 7])
def test_discretization_3box(dim, v0_given, v1_given):
    est = BoxDiscretization(
        dim=dim, n_boxes=3,
        v0=None if not v0_given else np.full((dim,), 0),
        v1=None if not v1_given else np.full((dim,), 1),
    )
    assert_equal(est.v0, [0]*dim if v0_given else None)
    assert_equal(est.v1, [1]*dim if v1_given else None)
    data = np.random.uniform(size=(20, dim))
    model = est.fit(data).fetch_model()
    dtraj = model.transform(data)
    traj_onehot = model.transform_onehot(data)

    for t in range(len(dtraj)):
        # check dtraj reflects spatially closest cluster center
        norms = np.linalg.norm(model.cluster_centers - data[t][None, :], axis=1)
        assert_equal(np.argmin(norms), dtraj[t])
        onehot_vec = np.zeros((model.n_clusters, ))
        onehot_vec[dtraj[t]] = 1.
        assert_equal(traj_onehot[t], onehot_vec)
