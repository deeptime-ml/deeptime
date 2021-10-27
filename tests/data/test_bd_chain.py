import pytest
from numpy.testing import assert_raises
from scipy.sparse import issparse

from deeptime.data import BirthDeathChain


@pytest.mark.parametrize("sparse", [True, False])
@pytest.mark.parametrize("q,p", [
    [[1.], [0.]],  # q must start with 0
    [[0., 1.], [0., 1.]],  # p must end on 0
    [[0., 1., 1.], [1., 1., 0.]]  # p + q must be <= 1. for all entries
])
def test_invalid_ctor_args(q, p, sparse):
    with assert_raises(ValueError):
        BirthDeathChain(q, p, sparse)


def assert_equal(x, y):
    from numpy.testing import assert_equal as impl
    if issparse(x):
        x = x.todense()
    if issparse(y):
        y = y.todense()
    impl(x, y)


def test_simple_bdc(sparse_mode):
    bdc = BirthDeathChain([0., 1.], [1., 0.], sparse=sparse_mode)
    assert_equal(bdc.p, [1., 0.])
    assert_equal(bdc.q, [0., 1.])
    assert_equal(bdc.dim, 2)
    assert_equal(bdc.transition_matrix, [[0, 1], [1, 0]])
    assert_equal(bdc.netflux(0, 1), [[0, 0.5], [0, 0]])
    assert_equal(bdc.netflux(1, 0), [[0, 0], [0.5, 0]])
    assert_equal(bdc.stationary_distribution, [.5, .5])
    assert_equal(bdc.rate(0, 1), 1.)
    assert_equal(bdc.rate(1, 0), 0.)
    assert_equal(bdc.totalflux(0, 1), .5)
    assert_equal(bdc.totalflux(1, 0), 0)
    assert_equal(bdc.committor_forward(0, 1), [0, 1])
    assert_equal(bdc.committor_forward(1, 0), [1, 0])
    assert_equal(bdc.committor_backward(0, 1), [1, 0])
    assert_equal(bdc.committor_backward(1, 0), [0, 1])
    assert_equal(bdc.flux(0, 1), [[0, 0.5], [0, 0]])
    assert_equal(bdc.flux(1, 0), [[0, 0], [0.5, 0]])
