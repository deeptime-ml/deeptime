from typing import Tuple

import numpy as np
from sklearn.pipeline import Pipeline

from ..base import Estimator, Model
from ..numeric import drop_nan_rows


class SINDy(Estimator):
    r"""
    TODO
    """

    def __init__(self):
        pass

    def fit(self, data: Tuple[np.ndarray, np.ndarray], **kwargs):
        x, x_dot = data[0], data[1]

        if x_dot is None:
            x_dot = np.gradient(x, axis=0)

        # Some differentiation methods produce nans near boundaries
        x, x_dot = drop_nan_rows(x, x_dot)

        steps = [("features", self.library), ("model", self.optimizer)]
        self.pipeline = Pipeline(steps)

        self.pipeline.fit(x, x_dot)

        self.n_input_features_ = self.pipeline.steps[0][1].n_input_features_
        self.n_output_features_ = self.pipeline.steps[0][1].n_output_features_

        self._model = SINDyModel(
            library=self.library, coefficients=self.optimizer.coef_
        )

        return self


class SINDyModel(Model):
    r"""TODO"""

    def __init__(self, library, coefficients):
        self.library = library
        self.coef_ = coefficients

    def transform(x):
        return self.library.transform(x)

    def predict(x):
        return np.dot(self.library.transform(x), self.coef_)
