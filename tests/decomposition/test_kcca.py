from numpy.testing import assert_equal

import deeptime as dt

def test_kcca_sanity():
    ds = dt.data.bickley_jet(200).endpoints_dataset()
    kernel = dt.kernels.GaussianKernel(.7)
    kcca = dt.decomposition.KernelCCA(kernel, n_eigs=5, epsilon=1e-3)
    kcca_model = kcca.fit((ds.data, ds.data_lagged)).fetch_model()

    assert_equal(kcca_model.eigenvalues.shape, (5,))
    assert_equal(kcca_model.eigenvectors.shape, (200, 5))
