import sktime.base as _base
import numpy as _np


class TICAModel(object):

    def __init__(self, *args, **kw):
        pass

    @property
    def cov(self):
        return 0

    @property
    def lagtime(self):
        return 1

    @property
    def cov_tau(self):
        return 0


class TICA(_base.Estimator):

    def __init__(self, lagtime, dim=None):
        self._model = TICAModel(lagtime, dim)

    @classmethod
    def from_model(cls, model: TICAModel):
        # do
        pass

    def fit(self, data, **kw):
        """

        Parameters
        ----------
        data: list of arrays

        Returns
        -------

        """
        if not (isinstance(data, (list, tuple)) and len(data) == 2 and len(data[0]) == len(data[1])):
            raise ValueError("Expected tuple of arrays of equal length!")

        n_splits = int(kw.get('n_splits', default=len(data[0]) // 100 if len(data[0]) >= 1e4 else 1))
        if 'n_splits' in kw.keys(): kw.pop('n_splits')

        for x, y in zip(_np.array_split(data[0], n_splits), _np.array_split(data[1], n_splits)):
            assert len(x) == len(y)
            if len(x) > 0:
                self.partial_fit((x, y), **kw)

        return self

    def partial_fit(self, partial_data, **kw):
        pass
        # update covariance matrices
        return self
