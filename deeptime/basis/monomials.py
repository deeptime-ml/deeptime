import numpy as np

from ._basis_bindings import evaluate_monomials as _eval
from .base import Observable


class Identity(Observable):
    r""" The identity. """

    def _evaluate(self, x):
        return x


class Monomials(Observable):
    r""" Monomial basis observable which transforms a number of n-dimensional
    datapoints :math:`\mathbf{x}\in\mathbb{R}^n` into (unique) monomials of at most degree :math:`p`.

    This means, that

    .. math::

        \mathbf{x} \mapsto \left\{ \prod_{d=1}^n \mathbf{x}_d^{k_d} : \sum k_d \leq p \right\}.

    The set is returned as a numpy ndarray of shape `(n_test_points, n_monomials)`, where `n_monomials` is the
    size of the set.

    Examples
    --------
    Given three test points in one dimension

    >>> import numpy as np
    >>> X = np.random.normal(size=(3, 1))

    Evaluating the monomial basis up to degree two yields :math:`x^0, x^1, x^2`, i.e., the expected shape is (3, 3)

    >>> Y = Monomials(p=2)(X)
    >>> Y.shape
    (3, 3)

    and, e.g., the second monomial of the third test point is the third test point itself:

    >>> np.testing.assert_almost_equal(Y[2, 1], X[2, 0])
    """

    def __init__(self, p: int):
        r""" Creates a new monomial basis of degree `p`.

        Parameters
        ----------
        p : int
            Maximum degree of the monomial basis. Must be positive.
        """
        assert p > 0
        self.p = p

    def _evaluate(self, x: np.ndarray):
        return _eval(self.p, x.T).T
