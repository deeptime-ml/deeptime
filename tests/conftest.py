# pytest specific configuration file containing eg fixtures.

import pytest
import numpy as np

#import warnings
#warnings.filterwarnings('error')


@pytest.fixture
def fixed_seed():
    state = np.random.mtrand.get_state()
    np.random.mtrand.seed(42)
    yield
    np.random.mtrand.set_state(state)
