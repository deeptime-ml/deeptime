import numpy as np
import pytest

from numpy.testing import *

from sktime.kernels import GaussianKernel, GeneralizedGaussianKernel, LaplacianKernel
from sktime.kernels.kernels import PolynomialKernel


@pytest.fixture
def data():
    X = np.random.normal(size=(50, 7))
    Y = np.random.normal(size=(30, 7))
    return X, Y


@pytest.mark.parametrize("kernel", [
    GaussianKernel(1.), GaussianKernel(2.), GaussianKernel(3.),
    GeneralizedGaussianKernel(np.linspace(3, 5, num=7)),
    LaplacianKernel(3.3),
    PolynomialKernel(3, 1.), PolynomialKernel(7, 3.3)
], ids=lambda k: str(k))
def test_consistency(data, kernel):
    xy_gram = kernel.apply(*data)
    assert_equal(xy_gram.shape, (50, 30))
    for i in range(50):
        for j in range(30):
            assert_almost_equal(xy_gram[i, j], kernel(data[0][i], data[1][j]))
