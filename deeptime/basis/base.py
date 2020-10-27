import numpy as np


class Observable(object):
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
