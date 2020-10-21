import numpy as np

import pytest

slepc4py = pytest.importorskip("slepc4py")
petsc4py = pytest.importorskip("petsc4py")

from sktime.numeric import slepc as slepc_impl


@pytest.mark.parametrize('method', ['lanczos', 'lapack', 'trlanczos', 'cross'])
def test_svd(method):
    X = np.random.normal(size=(50, 30))

    u, s, v = slepc_impl.svd(X, method=method)
    np.testing.assert_almost_equal(u.T @ np.diag(s) @ v, X)
