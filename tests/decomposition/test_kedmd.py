import numpy as np
from numpy.testing import assert_, assert_equal

import deeptime as dt


def test_sanity():
    data = dt.data.ellipsoids().observations(500)
    kernel = dt.kernels.GaussianKernel(1.)
    kedmd = dt.decomposition.KernelEDMD(kernel, n_eigs=4)
    model = kedmd.fit((data[:-1], data[1:])).fetch_model()
    phi = model.transform(data[:-1])
    assert_equal(phi.shape, (499, 4))
    assert_(np.abs(model.eigenvalues)[0] > 0)
    assert_(np.abs(model.eigenvalues)[1] > 0)
