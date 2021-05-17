from typing import Optional
from warnings import warn

import numpy as np
from scipy.integrate import odeint
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, ridge_regression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.validation import check_X_y

from ..base import Estimator, Model
from ..numeric import drop_nan_rows


class SINDyModel(Model):
    r"""The SINDy model. Stores the parameters learned by a :class:`SINDy` estimator
    to encode a first order differential equation model for the measurement data.
    It can be used to make derivative predictions, simulate forward in time from
    initial conditions, and for self-scoring.

    The model encodes a dynamical system

    .. math::
            \dot{X} = \Theta(X)\Xi

    via the following correpondences
    :code:`library` = :math:`\Theta` and
    :code:`(intercept, coefficients)` = :math:`\Xi^\top`.

    Parameters
    ----------
    library : library object
        The feature library, :math:`\Theta`.
        It is assumed that this object has already been fit to the input data.
        The object should implement  :meth:`transform`
        and :meth:`get_feature_names` methods.

    coefficients : np.ndarray, shape (n_input_features, n_output_features)
        Coefficients giving the linear combination of basis functions to
        approximate derivative data, i.e. :math:`\Xi^\top`. Note that
        :code:`coefficients` may or may not contain information about the intercepts
        depending on the library used (e.g. a polynomial library can contain the
        constant function).

    input_features : iterable of str
        List of input feature names.

    intercept : float, optional, default=0
        The intercept/bias for the learned model.
        If the library already contains a constant function, there is no
        need for an intercept term.
    """

    def __init__(self, library, coefficients, input_features, intercept=0):
        super().__init__()
        self.library = library
        self.coef_ = coefficients
        self.input_features = input_features
        self.intercept_ = intercept

    def transform(self, x):
        r"""Apply the functions of the feature library.

        This method computes :math:`\Theta(X)`.

        Parameters
        ----------
        x : np.ndarray, shape (n_samples, n_input_features)
            Measurement data.

        Returns
        -------
        y : np.ndarray, shape (n_samples, n_output_features)
            The feature library evaluated on :code:`x`,
            i.e. :math:`\Theta(X)`.
        """
        return self.library.transform(x)

    def predict(self, x):
        r"""Predict the derivative of :code:`x` using the learned model,
        i.e. predict :math:`\dot{X}` via :math:`\dot{X} \approx \Theta(X)\Xi`.

        Parameters
        ----------
        x : np.ndarray, shape (n_samples, n_input_features)
            Measurement data.

        Returns
        -------
        y : np.ndarray, shape (n_samples, n_input_features)
            Model prediction of the derivative :math:`\dot{X}`.
        """
        return np.matmul(self.transform(x), self.coef_.T) + self.intercept_

    def score(self, x, y=None, t=None, scoring=r2_score, **scoring_kws):
        r"""Compute a score for the time derivative prediction produced by the model.

        Parameters
        ----------
        x : np.ndarray, shape (n_samples, n_input_features)
            Measurement data.

        y : np.ndarray, shape (n_samples, n_input_features), optional, default=None
            Array of derivatives of :code:`x`.
            By default, :code:`np.gradient` is used to compute the derivatives.

        t : np.ndarray, shape (n_samples,) or a scalar, optional, default=None
            The times when the measurements in :code:`x` were taken or the
            (uniform) time spacing between measurements in :code:`x`.
            By default a timestep of 1 is assumed.
            This argument is ignored if :code:`y` is passed.

        scoring : callable, optional, default=r2_score
            Function by which to score the prediction.
            By default, the R^22 coefficient of determination is computed.
            See `Scikit-learn <https://scikit-learn.org/stable/modules/model_evaluation.html>`_
            for more options.

        Returns
        -------
        s : float
            Score for the time derivative prediction.
        """
        if y is None:
            if t is None:
                y = np.gradient(x, axis=0)
            else:
                y = np.gradient(x, t, axis=0)

        return scoring(y, self.predict(x), **scoring_kws)

    def simulate(self, x0, t, integrator=odeint, integrator_kws=None):
        """
        Simulate the SINDy model forward in time.

        Parameters
        ----------
        x0: numpy array, size [n_features]
            Initial condition from which to simulate.

        t: int or numpy array of size [n_samples]
            If the model is in continuous time, t must be an array of time
            points at which to simulate. If the model is in discrete time,
            t must be an integer indicating how many steps to predict.

        integrator: callable, optional (default :code:`odeint`)
            Function to use to integrate the system.
            Default is :code:`scipy.integrate.odeint`.

        integrator_kws: dict, optional (default None)
            Optional keyword arguments to pass to the integrator

        Returns
        -------
        x: numpy array, shape (n_samples, n_features)
            Simulation results.
        """

        if integrator_kws is None:
            integrator_kws = {}

        def rhs(x, _):
            return self.predict(x[np.newaxis, :])[0]

        return integrator(rhs, x0, t, **integrator_kws)

    def print(self, lhs=None, precision=3):
        r"""Print the learned equations in a human-readable way.

        Parameters
        ----------
        lhs : list of strings, optional, default=None
            List of variables to print on the left-hand side of the equations.
            By default :code:`self.input_features` are used.

        precision: int, optional, default=3
            Precision to be used when printing out model coefficients.
        """
        equations = self.equations(precision=precision)
        for i, eqn in enumerate(equations):
            if lhs:
                print(f"{lhs[i]}' = {eqn}")
            else:
                print(f"{self.input_features[i]}' = {eqn}")

    def equations(self, precision=3):
        """
        Get the right-hand sides of the learned equations.

        Parameters
        ----------
        precision : int, optional, default=3
            Precision to which coefficients are rounded.

        Returns
        -------
        equations : list of strings, length (n_input_features)
            List of model equations, with one for each input variable.
        """
        feature_names = self.library.get_feature_names(
            input_features=self.input_features
        )
        equation_list = [None] * len(self.coef_)

        for k, row_coef in enumerate(self.coef_):
            terms = [
                "{coef:.{precision}f} {fn}".format(coef=coef, precision=precision, fn=feature_names[i])
                for i, coef in enumerate(row_coef)
                if coef != 0
            ]
            equation_list[k] = " + ".join(terms)

        return equation_list

    @property
    def coefficients(self):
        r"""Returns the learned model coefficients, i.e. :math:`\Xi^\top`

        Returns
        -------
        coefficients : np.ndarray, shape (n_input_features, n_output_features)
            Learned model coefficients :math:`\Xi^\top`.
        """
        return self.coef_

    @property
    def intercept(self):
        r"""Returns the intercept (bias) for the learned model.

        Returns
        -------
        intercept : np.ndarray, shape (n_input_features,) or float
            The intercept or intercepts for each input feature.
        """
        return self.intercept_


class SINDy(Estimator):
    r"""Learn a dynamical systems model for measurement data using the
    Sparse Identification of Nonlinear Dynamical Systems (SINDy) method.

    For given measurement data :math:`X`, and a set of library functions evaluated
    on :math:`X`

    .. math::
        \Theta(X) = [\theta_1(X), \theta_2(X), \dots, \theta_k(X)],

    SINDy seeks a sparse set of coefficients :math:`\Xi` which satisfies

    .. math::
        \dot{X} \approx \Theta(X)\Xi.

    The i-th column of this matrix equation gives a differential equation for the
    i-th measurement variable (i-th column in :math:`X`). For more details see
    :footcite:`brunton2016sindy`.

    Parameters
    ----------
    library : library object, optional, default=None
        The candidate feature library, :math:`\Theta`.
        The object should implement a :meth:`fit`, :meth:`transform`,
        and :meth:`get_feature_names` methods. It should also have
        :attr:`n_input_features_` and :attr:`n_output_features_` attributes.
        By default a polynomial library of degree 2 is used.

    optimizer : optimizer object, optional, default=None
        The optimization routine used to solve the objective
        :math:`\dot{X} \approx \Theta(X)\Xi`.
        The object should have :meth:`fit` and :meth:`predict` methods
        and :attr:`coef_` and :attr:`intercept_` attributes. For example,
        any linear regressor from `sklearn.linear_model \
        <https://scikit-learn.org/stable/modules/linear_model.html>`_ should work.
        By default, :meth:`STLSQ` is used.

    input_features : list of strings, optional, default=None
        List of input feature names. By default, the names
        "x0", "x1", ..., "x{n_input_features}" is used.

    References
    ----------
    .. footbibliography::
    """

    def __init__(self, library=None, optimizer=None, input_features=None):
        super().__init__()
        if library is None:
            library = PolynomialFeatures(degree=2)
        if optimizer is None:
            optimizer = STLSQ(threshold=0.1)
        self.library = library
        self.optimizer = optimizer
        self.input_features = input_features

    def fit(self, x, y=None, t=None):
        r"""Fit the estimator to measurement data.

        Parameters
        ----------
        x : np.ndarray, shape (n_samples, n_input_features)
            Training/measurement data.
            Each row should correspond to one example and each column
            to a feature.

        y : np.ndarray, shape (n_samples, n_input_features), optional, default=None
            Array of derivatives of :code:`x`.
            By default, :code:`np.gradient` is used to compute the derivatives.

        t : np.ndarray, shape (n_samples,) or a scalar, optional, default=None
            The times when the measurements in :code:`x` were taken or the
            (uniform) time spacing between measurements in :code:`x`.
            By default a timestep of 1 is assumed.
            This argument is ignored if :code:`y` is passed.

        Returns
        -------
        self: SINDy
            Reference to self

        """
        if y is not None:
            x_dot = y
        elif t is not None:
            x_dot = np.gradient(x, t, axis=0)
        else:
            x_dot = np.gradient(x, axis=0)

        # Some differentiation methods produce nans near boundaries
        x, x_dot = drop_nan_rows(x, x_dot)

        steps = [("features", self.library), ("model", self.optimizer)]
        if hasattr(self.library, 'fit'):
            self.library.fit(x)

        self.n_input_features_ = x.shape[1]
        x_transformed = self.library.transform(x)
        self.n_output_features_ = x_transformed[1]
        self.optimizer.fit(x_transformed, x_dot)
        if self.input_features is None:
            self.input_features = [f"x{i}" for i in range(self.n_input_features_)]

        if hasattr(self.optimizer, "intercept_"):
            intercept = self.optimizer.intercept_
        else:
            intercept = 0

        self._model = SINDyModel(
            library=self.library,
            coefficients=self.optimizer.coef_,
            input_features=self.input_features,
            intercept=intercept,
        )

        return self

    def fetch_model(self) -> Optional[SINDyModel]:
        r""" Yields the latest model.

        Returns
        -------
        model : SINDyModel or None
            The model.
        """
        return super().fetch_model()


class STLSQ(LinearRegression):
    r"""Sequentially thresholded least squares algorithm.

    Attempts to minimize the objective function
    :math:`\|y - Xw\|^2_2 + \alpha \|w\|^2_2`
    by iteratively performing least squares and masking out
    elements of the weight that are below a given threshold.

    See this paper for more details :footcite:`brunton2016sindy`.

    Parameters
    ----------
    threshold : float, optional, default=0.1
        Minimum magnitude for a coefficient in the weight vector.
        Coefficients with magnitude below the threshold are set
        to zero.

    alpha : float, optional, default=0.05
        Optional L2 (ridge) regularization on the weight vector.

    max_iter : int, optional, default=20
        Maximum iterations of the optimization algorithm.

    ridge_kw : dict, optional, default=None
        Optional keyword arguments to pass to the ridge regression.

    fit_intercept : boolean, optional, default=False
        Whether to calculate the intercept for this model. If set to false, no
        intercept will be used in calculations.

    normalize : boolean, optional, default=False
        This parameter is ignored when :code:`fit_intercept` is set to False. If True,
        the regressors X will be normalized before regression by subtracting
        the mean and dividing by the l2-norm.

    copy_X : boolean, optional, default=True
        If True, X will be copied; else, it may be overwritten.

    References
    ----------
    .. footbibliography::
    """

    def __init__(self, threshold=0.1, alpha=0.05, max_iter=20, ridge_kw=None, normalize=False, fit_intercept=False,
                 copy_X=True):
        super().__init__(fit_intercept=fit_intercept, normalize=normalize, copy_X=copy_X)
        self.threshold = threshold
        self.alpha = alpha
        self.max_iter = max_iter
        self.ridge_kw = ridge_kw
        self.normalize = normalize
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X

    def fit(self, x_, y):
        r"""Fit to the data.

        Parameters
        ----------
        x_ : array-like, shape (n_samples, n_features)
            Training data (:math:`X` in the above equation).

        y : array-like, shape (n_samples,) or (n_samples, n_targets)
            Target values (:math:`y` in the above equation).

        Returns
        -------
        self : STLSQ
            Reference to self
        """
        # Do some preprocessing before fitting
        x_, y = check_X_y(x_, y, accept_sparse=[], y_numeric=True, multi_output=True)

        x, y, X_offset, y_offset, X_scale = self._preprocess_data(
            x_,
            y,
            fit_intercept=self.fit_intercept,
            normalize=self.normalize,
            copy=self.copy_X,
        )

        self.iters = 0
        self.ind_ = np.ones((y.shape[1], x.shape[1]), dtype=bool)
        self.coef_ = np.linalg.lstsq(x, y, rcond=None)[0].T  # initial guess
        self.history_ = [self.coef_]

        self._reduce(x, y)
        self.ind_ = np.abs(self.coef_) > 1e-14

        self._set_intercept(X_offset, y_offset, X_scale)
        return self

    def _reduce(self, x, y):
        r"""Iterates the thresholding. Assumes an initial guess is saved in
        self.coef_ and self.ind_
        """
        ind = self.ind_
        n_samples, n_features = x.shape
        n_targets = y.shape[1]
        n_features_selected = np.sum(ind)

        for _ in range(self.max_iter):
            if np.count_nonzero(ind) == 0:
                warn(
                    "Sparsity parameter is too big ({}) and eliminated all "
                    "coefficients".format(self.threshold)
                )
                coef = np.zeros((n_targets, n_features))
                break

            coef = np.zeros((n_targets, n_features))
            for i in range(n_targets):
                if np.count_nonzero(ind[i]) == 0:
                    warn(
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
            warn(
                "STLSQ._reduce did not converge after {} iterations.".format(
                    self.max_iter
                ),
                ConvergenceWarning,
            )
            try:
                coef
            except NameError:
                coef = self.coef_
                warn(
                    "STLSQ._reduce has no iterations left to determine coef",
                    ConvergenceWarning,
                )
        self.coef_ = coef
        self.ind_ = ind

    @staticmethod
    def _sparse_coefficients(dim, ind, coef, threshold):
        r"""Perform thresholding of the weight vector(s)"""
        c = np.zeros(dim)
        c[ind] = coef
        big_ind = np.abs(c) >= threshold
        c[~big_ind] = 0
        return c, big_ind

    def _regress(self, x, y):
        r"""Perform the ridge regression"""
        kw = self.ridge_kw or {}
        coef = ridge_regression(x, y, self.alpha, **kw)
        self.iters += 1
        return coef

    def _no_change(self):
        r"""Check if the coefficient mask has changed after thresholding"""
        this_coef = self.history_[-1].flatten()
        if len(self.history_) > 1:
            last_coef = self.history_[-2].flatten()
        else:
            last_coef = np.zeros_like(this_coef)
        return all(bool(i) == bool(j) for i, j in zip(this_coef, last_coef))
