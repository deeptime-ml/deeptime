# pytest specific configuration file containing eg fixtures.
import random

import pytest
import numpy as np
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


# import warnings
# warnings.filterwarnings('error')


@pytest.fixture
def fixed_seed():
    random.seed(42)
    np.random.mtrand.seed(42)
    fix_torch_seed(42)
