from typing import Tuple

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.validation import check_X_y

from ..base import Estimator, Model
from ..numeric import drop_nan_rows


class SINDy(Estimator):
    r"""
    TODO
    """

    def __init__(self, library=None, optimizer=None, input_features=None):
        if library is None:
            library = PolynomialFeatures(degree=2)
        if optimizer is None:
            optimizer = STLSQ(threshold=0.1)
        self.library = library
        self.optimizer = optimizer
        self.input_features = input_features

    def fit(self, data, **kwargs):
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

        if self.input_features is None:
            self.input_features = [f"x{i}" for i in range(self.n_input_features_)]

        self._model = SINDyModel(
            library=self.library,
            coefficients=self.optimizer.coef_,
            input_features=self.input_features,
        )

        return self


class SINDyModel(Model):
    r"""TODO"""

    def __init__(self, library, coefficients, input_features):
        self.library = library
        self.coef_ = coefficients
        self.input_features = input_features

    def transform(self, x):
        return self.library.transform(x)

    def predict(self, x):
        return np.dot(self.library.transform(x), self.coef_)

    def score(self, x, y=None, scoring=r2_score, **scoring_kws):
        if y is None:
            y = np.gradient(x, axis=0)

        return scoring(y, x, **scoring_kws)

    def print(self, lhs=None, precision=3):
        equations = self.equations(precision=precision)
        for i, eqn in enumerate(equations):
            if lhs:
                print(lhs[i] + "'", "=", eqn)
            else:
                print(self.input_features[i] + "'", "=", eqn)

    def equations(self, precision=3):
        """
        Get the right-hand side of the learned equations.
        """
        feature_names = self.library.get_feature_names(
            input_features=self.input_features
        )
        equation_list = [None] * len(self.coef_)

        for k, row_coef in enumerate(self.coef_):
            terms = [
                f"{self._round_terms(coef, precision)} {feature_names[k]}"
                for coef in row_coef
            ]
            equation_list[i] = "+".join(terms)

        return equation_list

    def _round_terms(coef, precision):
        if coef == 0:
            return ""
        else:
            return round(coef, precision)


class STLSQ(LinearRegression):
    r"""TODO"""

    def __init__(
        self,
        threshold=0.1,
        alpha=0.05,
        max_iter=20,
        ridge_kw=None,
        normalize=False,
        fit_intercept=False,
        copy_X=True,
    ):
        self.threshold = threshold
        self.alpha = alpha
        self.max_iter = max_iter
        self.ridge_kw = ridge_kw
        self.normalize = normalize
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X

    def fit(x_, y, sample_weight=None, **reduce_kws):
        # Do some preprocessing before fitting
        x_, y = check_X_y(x_, y, accept_sparse=[], y_numeric=True, multi_output=True)

        x, y, X_offset, y_offset, X_scale = self._preprocess_data(
            x_,
            y,
            fit_intercept=self.fit_intercept,
            normalize=self.normalize,
            copy=self.copy_X,
            sample_weight=sample_weight,
        )

        if sample_weight is not None:
            x, y = _rescale_data(x, y, sample_weight)

        self.iters = 0
        self.ind_ = np.ones((y.shape[1], x.shape[1]), dtype=bool)
        self.coef_ = np.linalg.lstsq(x, y, rcond=None)[0].T  # initial guess
        self.history_ = [self.coef_]

        self._reduce(x, y, **reduce_kws)
        self.ind_ = np.abs(self.coef_) > 1e-14

        self._set_intercept(X_offset, y_offset, X_scale)
        return self

    # def _reduce(self, x, y)
