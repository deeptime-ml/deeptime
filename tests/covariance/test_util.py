import numpy as np
import pytest
from numpy.testing import assert_equal

from deeptime.covariance.util.covar_c.covartools import variable_cols


@pytest.mark.parametrize("dtype", [np.int32, np.float32, np.bool, np.float64, np.complex, np.int64])
def test_variable_cols(dtype):
    const_data = np.full((500, 3), 0, dtype=dtype)
    assert_equal(variable_cols(const_data), [False, False, False])

    nonconst_data = const_data.copy()
    nonconst_data[0, 1] = 1

    assert_equal(variable_cols(nonconst_data), [False, True, False])
