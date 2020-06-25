import numpy as np
from numpy.testing import *

from sktime.numeric import is_diagonal_matrix, mdot


def test_is_diagonal_matrix():
    assert_(is_diagonal_matrix(np.diag([1, 2, 3, 4, 5])))
    assert_(not is_diagonal_matrix(np.array([[1, 2], [3, 4]])))


def test_mdot():
    A = np.random.normal(size=(5, 10))
    B = np.random.normal(size=(10, 20))
    C = np.random.normal(size=(20, 30))
    assert_almost_equal(mdot(A, B, C), A @ B @ C)
