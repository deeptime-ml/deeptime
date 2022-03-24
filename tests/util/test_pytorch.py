import torch

import pytest
from numpy.testing import assert_equal

from deeptime.util.torch import disableTF32

def test_disable_tf32():
    orig_tf32_setting = torch.backends.cuda.matmul.allow_tf32
    with disableTF32():
        assert_equal(torch.backends.cuda.matmul.allow_tf32,
                     False)
    assert_equal(torch.backends.cuda.matmul.allow_tf32,
                 orig_tf32_setting)
