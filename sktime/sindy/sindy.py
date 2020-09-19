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

    def _reduce(self, x, y):
        """Iterates the thresholding. Assumes an initial guess is saved in
        self.coef_ and self.ind_
        """
        ind = self.ind_
        n_samples, n_features = x.shape
        n_targets = y.shape[1]
        n_features_selected = np.sum(ind)

        for _ in range(self.max_iter):
            if np.count_nonzero(ind) == 0:
                warnings.warn(
                    "Sparsity parameter is too big ({}) and eliminated all "
                    "coefficients".format(self.threshold)
                )
                coef = np.zeros((n_targets, n_features))
                break

            coef = np.zeros((n_targets, n_features))
            for i in range(n_targets):
                if np.count_nonzero(ind[i]) == 0:
                    warnings.warn(
                        "Sparsity parameter is too big ({}) and eliminated all "
                        "coefficients".format(self.threshold)
                    )
                    continue
                coef_i = self._regress(x[:, ind[i]], y[:, i])
                coef_i, ind_i = self._sparse_coefficients(
                    n_features, ind[i], coef_i, self.threshold
                )
                coef[i] = coef_i
                ind[i] = ind_i

            self.history_.append(coef)
            if np.sum(ind) == n_features_selected or self._no_change():
                # could not (further) select important features
                break
        else:
            warnings.warn(
                "STLSQ._reduce did not converge after {} iterations.".format(
                    self.max_iter
                ),
                ConvergenceWarning,
            )
            try:
                coef
            except NameError:
                coef = self.coef_
                warnings.warn(
                    "STLSQ._reduce has no iterations left to determine coef",
                    ConvergenceWarning,
                )
        self.coef_ = coef
        self.ind_ = ind

    def _sparse_coefficients(self, dim, ind, coef, threshold):
        """Perform thresholding of the weight vector(s)"""
        c = np.zeros(dim)
        c[ind] = coef
        big_ind = np.abs(c) >= threshold
        c[~big_ind] = 0
        return c, big_ind

    def _regress(self, x, y):
        """Perform the ridge regression"""
        kw = self.ridge_kw or {}
        coef = ridge_regression(x, y, self.alpha, **kw)
        self.iters += 1
        return coef

    def _no_change(self):
        """Check if the coefficient mask has changed after thresholding"""
        this_coef = self.history_[-1].flatten()
        if len(self.history_) > 1:
            last_coef = self.history_[-2].flatten()
        else:
            last_coef = np.zeros_like(this_coef)
        return all(bool(i) == bool(j) for i, j in zip(this_coef, last_coef))
