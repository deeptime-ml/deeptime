import pytest
from numpy.testing import assert_array_almost_equal, assert_equal, assert_almost_equal, assert_raises

pytest.importorskip("torch")

import torch
import numpy as np
import deeptime.kernels as k


@pytest.fixture
def data():
    X = np.random.normal(size=(50, 7))
    Y = np.random.normal(size=(30, 7))
    return X, Y


@pytest.mark.parametrize("np_sigma", [False, True], ids=lambda x: f"npsigma={x}")
@pytest.mark.parametrize("np_data", [False, True], ids=lambda x: f"npdata={x}")
def test_consistency_gaussian(data, np_data, np_sigma):
    sigma = 1.1
    k2 = k.GaussianKernel(sigma)
    if not np_sigma:
        sigma = torch.from_numpy(np.array(sigma))
    k1 = k.TorchGaussianKernel(sigma)

    x, y = data[0], data[1]

    with assert_raises(ValueError):
        k1.apply(data[0], torch.from_numpy(data[1]))

    if not np_data:
        x, y = torch.from_numpy(x), torch.from_numpy(y)
    with torch.no_grad():
        out1 = k1.apply(x, y)

    out2 = k2.apply(*data)

    assert_array_almost_equal(out1, out2)


@pytest.mark.parametrize("np_sigma", [False, True], ids=lambda x: f"npsigma={x}")
@pytest.mark.parametrize("np_data", [False, True], ids=lambda x: f"npdata={x}")
def test_call_gaussian(data, np_data, np_sigma):
    sigma = 1.1
    kernel_ref = k.GaussianKernel(sigma)

    if not np_sigma:
        sigma = torch.from_numpy(np.array(sigma))
    kernel = k.TorchGaussianKernel(sigma)
    x, y = data[0][:15], data[1][:15]
    if not np_data:
        x, y = torch.from_numpy(x), torch.from_numpy(y)
    kxy = kernel(x, y)

    assert_equal(kxy.shape, (15,))
    for i in range(15):
        assert_almost_equal(kxy[i], kernel_ref(data[0][i], data[1][i]))
