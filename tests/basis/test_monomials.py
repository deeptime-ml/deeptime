import numpy as np
import pytest
from numpy.testing import assert_equal, assert_almost_equal

from sktime.basis.monomials import Monomials

import operator as op
from functools import reduce


def ncr(n, r):
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer // denom


@pytest.mark.parametrize("degree", [2, 4, 5], ids=lambda x: f"degree={x}")
@pytest.mark.parametrize('state_space_dim', [5, 6], ids=lambda x: f"state_space_dim={x}")
@pytest.mark.parametrize('n_test_points', [3, 4], ids=lambda x: f"n_test_points={x}")
def test_eval(degree, state_space_dim, n_test_points):
    x = np.random.normal(size=(state_space_dim, n_test_points))
    y1 = Monomials(degree)(x)
    assert_equal(y1.shape, (ncr(degree + x.shape[0], x.shape[0]), n_test_points))

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
