import abc
import warnings

import numpy as np

from sktime.base import Estimator, Model


# TODO: copy over?


class LaggedModelValidation(Model):
    def __init__(self, estimates=None, estimates_conf=None, predictions=None, predictions_conf=None, lagtimes=None):
        self._estimates = estimates
        self._estimates_conf = estimates_conf
        self._predictions = predictions
        self._predictions_conf = predictions_conf
        self._lagtimes = lagtimes

    @property
    def estimates(self):
        """ Returns estimates at different lagtimes

        Returns
        -------
        Y : ndarray(T, n)
            each row contains the n observables computed at one of the T lagtimes.
        """
        return self._estimates

    @property
    def estimates_conf(self):
        """ Returns the confidence intervals of the estimates at different
        lagtimes (if available).

        If not available, returns None.

        Returns
        -------
        estimates_conf : ndarray(T, n, k)
            each row contains the lower confidence bound of n observables
            computed at one of the T lag times.

        """
        return self._estimates_conf

    @property
    def predictions(self):
        """ Returns tested model predictions at different lagtimes

        Returns
        -------
        Y : ndarray(T, n)
            each row contains the n observables predicted at one of the T lag
            times by the tested model.

        """
        return self._predictions

    @property
    def predictions_conf(self):
        """ Returns the confidence intervals of the estimates at different
        lagtimes (if available)

        If not available, returns None.

        Returns
        -------
        L : ndarray(T, n)
            each row contains the lower confidence bound of n observables
            computed at one of the T lag times.

        R : ndarray(T, n)
            each row contains the upper confidence bound of n observables
            computed at one of the T lag times.

        """
        return self._predictions_conf


class LaggedModelValidator(Estimator, metaclass=abc.ABCMeta):
    r""" Validates a model estimated at lag time tau by testing its predictions
    for longer lag times

    Parameters
    ----------
    test_model : Model
        Model to be tested

    test_estimator : Estimator
        Parametrized Estimator that has produced the model

    mlags : int or int-array, default=10
        multiples of lag times for testing the Model, e.g. range(10).
        A single int will trigger a range, i.e. mlags=10 maps to
        mlags=range(1, 10). The setting None will choose mlags automatically
        according to the longest available trajectory

    conf : float, default = 0.95
        confidence interval for errors

    err_est : bool, default=False
        if the Estimator is capable of error calculation, will compute
        errors for each tau estimate. This option can be computationally
        expensive.

    """
    def __init__(self, test_model, test_estimator, mlags=None, conf=0.95, err_est=False):
        # set model and estimator
        self.test_model = test_model
        import copy
        self.test_estimator = copy.deepcopy(test_estimator)

        try:
            input_lagtime = self.test_estimator.lagtime
        except AttributeError:
            try:
                input_lagtime = self.test_model.lagtime
            except AttributeError:
                raise RuntimeError(
                    'Neither provided model nor estimator provides the "lagtime" attribute. Cannot proceed.')
        self.input_lagtime = input_lagtime

        # set conf and error handling
        self.conf = conf
        self.has_errors = hasattr(test_model, 'samples')
        self.mlags = mlags

        self.err_est = err_est
        if err_est and not self.has_errors:
            raise ValueError('Requested errors on the estimated models, '
                             'but the model is not able to calculate errors at all')

        super(LaggedModelValidator, self).__init__()

    def _set_mlags(self, data, lagtime: int):
        # set mlags, we do it in fit, so we can obtain the maximum possible lagtime from the trajectory data.
        from numbers import Integral
        if not isinstance(data, list):
            data = [data]

        mlags = self.mlags
        maxlength = np.max([len(x) for x in data])
        maxmlag = int(np.floor(maxlength / lagtime))
        if mlags is None:
            mlags = maxmlag
        elif isinstance(mlags, Integral):
            mlags = np.arange(1, mlags)
        mlags = np.asarray(mlags, dtype='i')
        if np.any(mlags > maxmlag):
            mlags = mlags[np.where(mlags <= maxmlag)]
            warnings.warn('Dropped lag times exceeding data lengths')
        if np.any(mlags < 0):
            mlags = mlags[np.where(mlags >= 0)]
            warnings.warn('Dropped negative multiples of lag time')

        if any(x <= 0 for x in mlags):
            raise ValueError('multiples of lagtimes have to be greater zero.')
        self.mlags = mlags

    def _create_model(self) -> LaggedModelValidation:
        return LaggedModelValidation()

    def fit(self, data):
        # set lag times
        self._set_mlags(data, self.input_lagtime)
        lags = self.mlags * self.input_lagtime

        predictions = []
        predictions_conf = []
        estimates = []
        estimates_conf = []
        estimated_models = []

        # estimate models at multiple of input lag time.
        for lag in lags:
            self.test_estimator.lagtime = lag
            self.test_estimator.fit(data)
            estimated_models.append(self.test_estimator.fetch_model())

        for mlag in self.mlags:
            # make a prediction using the test model
            predictions.append(self._compute_observables(self.test_model, mlag=mlag))
            # compute prediction errors if we can
            if self.has_errors:
                l, r = self._compute_observables_conf(self.test_model, mlag=mlag)
                predictions_conf.append((l, r))

        for model in estimated_models:
            if model is None: continue
            # evaluate the estimate at lagtime*mlag
            estimates.append(self._compute_observables(model))
            if self.has_errors and self.err_est:
                l, r = self._compute_observables_conf(model)
                estimates_conf.append((l, r))

        # build arrays
        estimates = np.array(estimates)
        predictions = np.array(predictions)
        if self.has_errors:
            predictions_conf = np.array(predictions_conf)
        else:
            predictions_conf = np.array((None, None))
        if self.has_errors and self.err_est:
            estimates_conf = np.array(estimates_conf)
        else:
            estimates_conf = np.array((None, None))

        self._model.__init__(estimates=estimates, estimates_conf=estimates_conf,
                             predictions=predictions, predictions_conf=predictions_conf, lagtimes=lags)

        return self

    @abc.abstractmethod
    def _compute_observables(self, model, mlag=1):
        """Compute observables for given model

        Parameters
        ----------
        model : Model
            model to compute observable for.

        mlag : int, default=1
            if 1, just compute the observable for given model. If not 1, use
            model to predict result at multiple of given model lagtime.

        Returns
        -------
        Y : ndarray
            array with results

        """
        pass

    @abc.abstractmethod
    def _compute_observables_conf(self, model, mlag=1, conf=0.95):
        """Compute confidence interval for observables for given model

        Parameters
        ----------
        model : Model
            model to compute observable for. model can be None if mlag=0.
            This scenario must be handled.

        mlag : int, default=1
            if 1, just compute the observable for given model. If not 1, use
            model to predict result at multiple of given model lagtime.

        Returns
        -------
        L : ndarray
            array with lower confidence bounds
        R : ndarray
            array with upper confidence bounds

        """
        pass
