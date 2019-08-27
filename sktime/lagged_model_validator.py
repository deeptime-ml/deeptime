import abc

import numpy as np

from sktime.base import Estimator, Model
# TODO: copy over?
from sklearn.model_selection import ParameterGrid


class LaggedModelValidation(Model):
    def __init__(self, estimates=None, estimates_conf=None, predictions=None, predictions_conf=None):
        self._estimates = estimates
        self._estimates_conf = estimates_conf
        self._predictions = predictions
        self._predictions_conf = predictions_conf

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
        return self._estimates_conf,

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


class LaggedModelValidator(Estimator):
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
        assert hasattr(test_estimator, 'lagtime')
        self.test_estimator = test_estimator
        # set conf and error handling
        self.conf = conf
        self.has_errors = hasattr(test_model, 'samples')
        #if self.has_errors:
        #    self.test_model.set_model_params(conf=conf)

        self.mlags = mlags

        self.err_est = err_est
        if err_est and not self.has_errors:
            raise ValueError('Requested errors on the estimated models, '
                             'but the model is not able to calculate errors at all')

        super(LaggedModelValidator, self).__init__()

    def _get_mlags(self, data, lagtime: int, mlags=None):
        # set mlags
        maxlength = np.max([len(x) for x in data])
        import math
        maxmlag = int(math.floor(maxlength / lagtime))
        if mlags is None:
            mlags = maxmlag
        from numbers import Integral
        if isinstance(mlags, Integral):
            mlags = np.arange(1, mlags)
        mlags = mlags.astype(dtype='i')
        if np.any(mlags > maxmlag):
            mlags = mlags[np.where(mlags <= maxmlag)]
        if np.any(mlags < 0):
            mlags = mlags[np.where(mlags >= 0)]

        assert 0 not in mlags
        return mlags

    def _create_model(self) -> LaggedModelValidation:
        return LaggedModelValidation()

    def fit(self, data, mlags=10):
        # lag times
        mlags = self._get_mlags(data, self.test_estimator.lagtime, mlags)
        assert isinstance(mlags, np.ndarray) and mlags.dtype == 'i'
        lags = mlags * self.test_estimator.lagtime
        pargrid = ParameterGrid({'lagtime': lags})
        # do we have zero lag? this must be treated separately
        include0 = mlags[0] == 0
        if include0:
            pargrid = pargrid[1:]

        predictions = []
        predictions_conf = []

        estimates = []
        estimates_conf = []

        # run estimates
        estimated_models = []
        for p in pargrid:
            self.test_estimator.lagtime = p
            self.test_estimator.fit(data)
            estimated_models.append(self.test_estimator.fetch_model())

        if include0:
            estimated_models.insert(0, None)

        for i, mlag in enumerate(mlags):
            if mlag == 0:
                continue
            # make a prediction using the current model
            predictions.append(self._compute_observables(self.test_model, mlag=mlag))
            # compute prediction errors if we can
            if self.has_errors:
                l, r = self._compute_observables_conf(self.test_model, mlag=mlag)
                predictions_conf.append((l, r))

            # do an estimate at this lagtime
            model = estimated_models[i]
            estimates.append(self._compute_observables(model))
            if self.has_errors and self.err_est:
                l, r = self._compute_observables_conf(model)
                estimates_conf.append((l, r))
        # TODO: fill
        if include0:
            dummy = np.ones_like(predictions[-1]) * np.nan
            predictions.insert(0, dummy)
            if self.has_errors:
                dummy = np.ones_like(predictions_conf[-1]) * np.nan
                predictions_conf.insert()

        # build arrays
        estimates = np.array(estimates)
        predictions = np.array(predictions)
        if self.has_errors:
            predictions_conf = np.array(predictions_conf)
        else:
            predictions_conf = None
        if self.has_errors and self.err_est:
            estimates_conf = np.array(estimates_conf)
        else:
            estimates_conf = None

        m = self._model
        m.estimates = estimates
        m.estimates_conf = estimates_conf
        m.predictions = predictions
        m.predictions_conf = predictions_conf

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
        raise NotImplementedError('_compute_observables is not implemented. Must override it in subclass!')

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
        raise NotImplementedError('_compute_observables is not implemented. Must override it in subclass!')
