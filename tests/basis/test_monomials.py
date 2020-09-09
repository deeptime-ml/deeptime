import numpy as np
from sktime.basis.monomials import Monomials


def test_eval():
    x = np.random.normal(size=(1, 1))
    y1 = Monomials(8)(x)

    out = np.array([[
        x[0, 0]**i for i in range(9)
    ]])
    np.testing.assert_array_almost_equal(y1.squeeze(), out.squeeze())
