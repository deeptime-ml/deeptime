import numpy as np


class Observable(object):

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
        x : (d, N) np.ndarray
            Evaluates the observable for N d-dimensional data points.

        Returns
        -------
        out : (p, N) np.ndarray
            Result of the evaluation for each data point.
        """
        return self._evaluate(x)
