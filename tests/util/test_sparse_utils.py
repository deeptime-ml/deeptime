import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.sparse import csr_matrix

from deeptime.util.sparse import remove_negative_entries


def test_remove_negative_entries():
    mat = np.random.randn(10, 10)
    mat_plus = remove_negative_entries(csr_matrix(mat))
    assert_array_almost_equal(mat_plus.toarray(), np.maximum(0., np.copy(mat)))
