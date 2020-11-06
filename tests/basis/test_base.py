import numpy as np
from numpy.testing import assert_raises, assert_equal

import deeptime as dt


def test_identity():
    o = dt.basis.Identity()
    args = np.array([5., 5.])
    assert_equal(o(args), args)


def test_interface():
    o = dt.basis.Observable()
    with assert_raises(NotImplementedError):
        o(np.random.normal())
