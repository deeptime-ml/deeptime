from typing import List

import numpy as np

from ._basis_bindings import evaluate_monomials as _eval, power_matrix as _power_matrix, feature_names as _fn
from ._base import Observable


class Identity(Observable):
    r""" The identity. """

    def _evaluate(self, x):
        return x

    @staticmethod
    def get_feature_names(input_features=None):
        if input_features is None:
            return ['x']
        else:
            assert len(input_features) == 1
            return input_features


class Monomials(Observable):
    r""" Monomial basis observable which transforms a number of d-dimensional
    datapoints :math:`\mathbf{x}\in\mathbb{R}^d` into (unique) monomials of at most degree :math:`p`.

    This means, that

    .. math::

        \mathbf{x} \mapsto \left\{ \prod_{d=1}^n \mathbf{x}_d^{k_d} : \sum k_d \leq p \right\}.

    The set is returned as a numpy ndarray of shape `(n_test_points, n_monomials)`, where `n_monomials` is the
    size of the set.

    Parameters
    ----------
    p : int
        Maximum degree of the monomial basis. Must be positive.
    d : int
        The dimension of the input.

    Examples
    --------
    Given three test points in one dimension

    >>> import numpy as np
    >>> X = np.random.normal(size=(3, 1))

    Evaluating the monomial basis up to degree two yields :math:`x^0, x^1, x^2`, i.e., the expected shape is (3, 3)

    >>> Y = Monomials(p=2, d=1)(X)
    >>> Y.shape
    (3, 3)

    and, e.g., the second monomial of the third test point is the third test point itself:

    >>> np.testing.assert_almost_equal(Y[2, 1], X[2, 0])
    """

    def __init__(self, p: int, d: int):
        from scipy.special import binom

        assert p > 0
        self.p = p
        self.d = d
        self._n_monomials = int(binom(self.p + self.d, self.p))
        self._power_matrix = _power_matrix(self.d, self._n_monomials)

    def _evaluate(self, x: np.ndarray):
        if self.d is not None and x.shape[1] != self.d:
            raise ValueError(f"Input had the wrong dimension {x.shape[1]}, this basis requires {self.d}.")
        return _eval(self.p, x.T, self._power_matrix).T

    def get_feature_names(self, input_features=None) -> List[str]:
        r""" Yields a list of feature names, optionally given input feature names.

        Parameters
        ----------
        input_features : list of str, optional, default=None
            If not `None`, replaces the input feature names.

        Returns
        -------
        feature_names : list of str
            Feature names corresponding to each monomial.
        """
        if input_features is None:
            input_features = [f'x{i}' for i in range(self.d)]
        return _fn(input_features, self._power_matrix)
