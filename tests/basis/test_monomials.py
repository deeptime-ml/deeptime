import numpy as np
import pytest
from numpy.testing import assert_equal, assert_almost_equal, assert_, assert_raises

from deeptime.basis import Monomials, Identity

import operator as op
from functools import reduce


def ncr(n, r):
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer // denom


@pytest.mark.parametrize('input_features', [None, ['y']])
def test_feature_names(input_features):
    mon = Monomials(3, 1)
    feature_names = mon.get_feature_names(input_features=input_features)
    feat = 'x0' if input_features is None else input_features[0]
    assert_('' in feature_names)
    assert_(f'{feat}' in feature_names)
    assert_(f'{feat}^2' in feature_names)
    assert_(f'{feat}^3' in feature_names)

    identity = Identity()
    assert_("z" in identity.get_feature_names(["z"]))
    assert_("x" in identity.get_feature_names())


@pytest.mark.parametrize("degree", [2, 4, 5], ids=lambda x: f"degree={x}")
@pytest.mark.parametrize('state_space_dim', [5, 6], ids=lambda x: f"state_space_dim={x}")
@pytest.mark.parametrize('n_test_points', [3, 4], ids=lambda x: f"n_test_points={x}")
def test_eval(degree, state_space_dim, n_test_points):
    x = np.random.normal(size=(state_space_dim, n_test_points))
    y1 = Monomials(degree, state_space_dim)(x.T).T
    assert_equal(y1.shape, (ncr(degree + x.shape[0], x.shape[0]), n_test_points))

    with assert_raises(ValueError):
        # state space dim mismatch
        Monomials(degree, state_space_dim)(np.random.normal(size=(n_test_points, state_space_dim+1)))

    # checks that for each test point x there are monomial evaluations
    # x_i^{p1} * x_j^{p2} with p1 + p2 <= degree
    for test_point_index in range(n_test_points):
        y = y1[:, test_point_index]

        for pi in x[:, test_point_index]:
            for pj in x[:, test_point_index]:
                for i in range(degree + 1):
                    for j in range(degree + 1):
                        if i + j <= degree:
                            xx = pi ** i * pj ** j
                            mindist = np.min(np.abs(y - xx))
                            assert_almost_equal(mindist, 0)
