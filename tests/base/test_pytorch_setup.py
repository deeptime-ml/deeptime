import pytest
from numpy.testing import assert_array_almost_equal

pytest.importorskip("torch")

import numpy as np
import torch


def test_pytorch_installation():
    r""" Fast-failing test that can be used to predetermine faulty pytorch installations. Can save some CI time. """
    X = torch.from_numpy(np.random.normal(size=(20, 10)))
    Y = torch.from_numpy(np.random.normal(size=(10, 20)))
    M = X @ Y
    with torch.no_grad():
        assert_array_almost_equal(X.numpy() @ Y.numpy(), M.numpy())
