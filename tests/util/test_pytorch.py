import pytest
torch = pytest.importorskip('torch')

from numpy.testing import assert_equal

import torch

from deeptime.util.torch import disable_TF32

def test_disable_tf32():
    orig_tf32_setting = torch.backends.cuda.matmul.allow_tf32
    with disable_TF32():
        assert_equal(torch.backends.cuda.matmul.allow_tf32,
                     False)
    assert_equal(torch.backends.cuda.matmul.allow_tf32,
                 orig_tf32_setting)
