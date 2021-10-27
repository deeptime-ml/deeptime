import numpy as np
import pytest
from numpy.testing import assert_equal, assert_almost_equal, assert_raises

import deeptime as dt
from deeptime.kernels import GaussianKernel, GeneralizedGaussianKernel, LaplacianKernel, PolynomialKernel


@pytest.fixture
def data():
    X = np.random.normal(size=(50, 7))
    Y = np.random.normal(size=(30, 7))
    return X, Y


def test_base():
    with assert_raises(NotImplementedError):
        dt.kernels.Kernel()(1, 2)  # base class does not implement call operator
    k = GaussianKernel(1.)
    with assert_raises(ValueError):
        k * "weird object"  # product kernel only supported with other kernels


@pytest.mark.parametrize("kernel", [
    GaussianKernel(1.), GaussianKernel(2.), GaussianKernel(3.),  # some Gaussian kernels
    GaussianKernel(1., impl='binomial'), GaussianKernel(3., impl='binomial'),  # some Gaussian kernels, binomial impl
    GeneralizedGaussianKernel(np.linspace(3, 5, num=7)),  # a generalized Gaussian kernel
    LaplacianKernel(3.3),  # a Laplacian kernel
    PolynomialKernel(3, 1.), PolynomialKernel(7, 3.3),  # some polynomial kernels
    PolynomialKernel(3, 1.) * GaussianKernel(3.) * LaplacianKernel(5.),  # product kernel
    PolynomialKernel(3, 1.) * GeneralizedGaussianKernel(np.linspace(3, 5, num=7)) * LaplacianKernel(5.)  # product k.
], ids=lambda k: str(k))
def test_consistency(data, kernel):
    xy_gram = kernel.apply(*data)
    assert_equal(xy_gram.shape, (50, 30))
    for i in range(50):
        for j in range(30):
            assert_almost_equal(xy_gram[i, j], kernel(data[0][i], data[1][j]), decimal=5)
