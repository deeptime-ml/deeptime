# pytest specific configuration file containing eg fixtures.
import os
import random

import numpy as np
import pytest

from deeptime.util.platform import module_available

if module_available("torch"):
    import torch


    def fix_torch_seed(seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
else:
    def fix_torch_seed(_):
        pass


@pytest.fixture
def fixed_seed():
    random.seed(42)
    np.random.mtrand.seed(42)
    fix_torch_seed(42)
    yield
    new_seed = int.from_bytes(os.urandom(16), 'big') % (2 ** 32 - 1)
    random.seed(new_seed)
    np.random.mtrand.seed(new_seed)
    fix_torch_seed(new_seed)


@pytest.fixture(params=[False, True], ids=lambda x: f"{'sparse' if x else 'dense'}")
def sparse_mode(request):
    yield request.param

