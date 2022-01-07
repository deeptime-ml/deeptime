import numpy as np
from deeptime.numeric import logsumexp

def test_logsumexp():
    array = np.random.rand(5)
    res = logsumexp(array)
    np.testing.assert_almost_equal(res, np.log(np.sum(np.exp(array))))