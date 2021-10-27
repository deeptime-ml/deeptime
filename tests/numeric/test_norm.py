import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_

from deeptime.numeric import schatten_norm


@pytest.mark.parametrize('hermitian', [False, True])
def test_schatten_norm(hermitian):
    mat = np.random.normal(size=(15, 15))
    if hermitian:
        mat = mat @ mat.T
    assert_array_almost_equal(np.linalg.norm(mat, ord='nuc'), schatten_norm(mat, 1, hermitian=hermitian))
    assert_array_almost_equal(np.linalg.norm(mat, ord='fro'), schatten_norm(mat, 2, hermitian=hermitian))
    if hermitian:
        assert_array_almost_equal(schatten_norm(mat, 3.3, hermitian=True), schatten_norm(mat, 3.3, hermitian=False))

    for i in range(1, 15):
        n1 = schatten_norm(mat, i, hermitian=hermitian)
        n2 = schatten_norm(mat, i + 0.9, hermitian=hermitian)
        n3 = schatten_norm(mat, i + 1.8, hermitian=hermitian)
        assert_(n1 >= n2 >= n3)
