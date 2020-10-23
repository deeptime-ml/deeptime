import numpy as np


class Observable(object):

    def _evaluate(self, x):
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
