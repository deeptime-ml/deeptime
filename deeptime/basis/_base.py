from typing import Callable

import numpy as np
from ..base import Transformer


class Observable(Transformer):
    r""" An object that transforms a series of state vectors :math:`X\in\mathbb{R}^{T\times n}` into a
    series of observables :math:`\Psi(X)\in\mathbb{R}^{T\times k}`, where `n` is the dimension of each state vector
    and `k` the dimension of the observable space.
    """

    def _evaluate(self, x: np.ndarray):
        r""" Evalues the observable on input data `x`.

        Parameters
        ----------
        x : (T, n) ndarray
            Input data.

        Returns
        -------
        y : (T, m) ndarray
            Basis applied to input data.
        """
        raise NotImplementedError()

    def __call__(self, x: np.ndarray):
        r""" Evaluation of the observable.

        Parameters
        ----------
        x : (N, d) np.ndarray
            Evaluates the observable for N d-dimensional data points.

        Returns
        -------
        out : (N, p) np.ndarray
            Result of the evaluation for each data point.
        """
        return self._evaluate(x)

    def transform(self, data, **kwargs):
        return self(data)


class Concatenation(Observable):
    r"""Concatenation operation to evaluate :math:`(f_1 \circ f_2)(x) = f_1(f_2(x))`, where
    :math:`f_1` and :math:`f_2` are observables.

    Parameters
    ----------
    obs1 : Callable
        First observable :math:`f_1`.
    obs2 : Callable
        Second observable :math:`f_2`.
    """

    def __init__(self, obs1: Callable[[np.ndarray], np.ndarray], obs2: Callable[[np.ndarray], np.ndarray]):
        self.obs1 = obs1
        self.obs2 = obs2

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        return self.obs1(self.obs2(x))
