import numpy as np
from deeptime.numeric import logsumexp

def test_logsumexp():
    array = np.random.rand(5)
    res = logsumexp(array)
    assert res == np.log(np.sum(np.exp(array)))