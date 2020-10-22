import pytest
from numpy.testing import assert_array_almost_equal, assert_equal, assert_almost_equal

pytest.importorskip("torch")

import torch
import numpy as np
import deeptime.kernels as k


@pytest.fixture
def data():
    X = np.random.normal(size=(50, 7))
    Y = np.random.normal(size=(30, 7))
    return X, Y


def test_consistency_gaussian(data):
    k1 = k.TorchGaussianKernel(1.1)
    k2 = k.GaussianKernel(1.1)

    with torch.no_grad():
        out1 = k1.apply(torch.from_numpy(data[0]), torch.from_numpy(data[1])).numpy()

    out2 = k2.apply(*data)

    assert_array_almost_equal(out1, out2)


def test_call_gaussian(data):
    kernel = k.TorchGaussianKernel(1.1)
    kernel_ref = k.GaussianKernel(1.1)
    kxy = kernel(torch.from_numpy(data[0][:15]), torch.from_numpy(data[1][:15]))

    assert_equal(kxy.shape, (15,))
    for i in range(15):
        assert_almost_equal(kxy[i], kernel_ref(data[0][i], data[1][i]))
