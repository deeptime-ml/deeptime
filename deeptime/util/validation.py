import warnings

import numpy as np

from threadpoolctl import threadpool_limits

from ..base import Estimator, Model, Observable, BayesianModel
from .parallel import handle_n_jobs, joining, multiprocessing_context
from .platform import handle_progress_bar


def _imap_wrapper(args):
    with threadpool_limits(limits=1, user_api='blas'):
        i, fun, arguments = args
        result = fun(*arguments)
        return i, result


class LaggedModelValidation(Model):
    def __init__(self, estimates=None, estimates_samples=None, predictions=None, predictions_samples=None,
                 lagtimes=None):
        r""" Result of a lagged model validator.

        Parameters
        ----------
        estimates : list, optional, default=None
            Estimated values for each lagtime.
        estimates_samples : list, optional, default=None
            Samples around estimated values for each lagtime.
        predictions : list, optional, default=None
            Predicted values based on model propagation.
        predictions_samples : list, optional, default=None
            Samples around predicted values.
        lagtimes : list, optional, default=None
            The lagtimes.
        """
        self._estimates = np.array(estimates)
        self._estimates_samples = np.array(estimates_samples) if estimates_samples is not None else None
        self._predictions = np.array(predictions)
        self._predictions_samples = np.array(predictions_samples) if predictions_samples is not None else None
        self._lagtimes = lagtimes

    @property
    def n_components(self):
        return len(self.estimates[0]) if self.estimates is not None and len(self.estimates) > 0 else 0

    @property
    def lagtimes(self):
        r""" Lagtimes at which estimations and predictions were performed. """
        return self._lagtimes

    @property
    def estimates(self):
        """ Estimates at different lagtimes.

        :getter: each row contains the n observables computed at one of the T lagtimes.
        :type: ndarray(T, n, n)
        """
        return self._estimates

    @property
    def estimates_samples(self):
        r""" Returns the sampled estimates, i.e., a list of arrays as described in :attr:`estimates`. Can be `None`. """
        return self._estimates_samples

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
    def predictions_samples(self):
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
        return self._predictions_samples

    @property
    def has_errors(self):
        return self.predictions_samples is not None and len(self.predictions_samples) > 0

    @property
    def err_est(self):
        return self.estimates_samples is not None


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
        mlags=range(1, 10).
    err_est : bool, default=False
        if the Estimator is capable of error calculation, will compute
        errors for each tau estimate. This option can be computationally
        expensive.
    """

    def __init__(self, test_model: Model, test_estimator: Estimator, test_model_lagtime: int, mlags, err_est=False):
        # set model and estimator
        self._test_model = test_model
        import copy
        self.test_estimator = copy.deepcopy(test_estimator)

        self.input_lagtime = test_model_lagtime

        self.has_errors = hasattr(test_model, 'samples')
        self._set_mlags(mlags)

        self.err_est = err_est
        if err_est and not self.has_errors:
            raise ValueError('Requested errors on the estimated models, '
                             'but the model is not able to calculate errors at all')

        super(LaggedModelValidator, self).__init__()

    def _set_mlags(self, mlags):
        # set mlags, we do it in fit, so we can obtain the maximum possible lagtime from the trajectory data.
        from numbers import Integral
        if isinstance(mlags, Integral):
            mlags = np.arange(mlags)
        mlags = np.asarray(mlags, dtype=int)

        mlags = np.atleast_1d(mlags)
        if (mlags < 0).any():
            raise ValueError('multiples of lagtimes have to be greater equal zero.')
        self.mlags = mlags

    def _effective_mlags(self, data, lagtime):
        if not isinstance(data, list):
            data = [data]
        maxlength = np.max([len(x) for x in data])
        maxmlag = int(np.floor(maxlength / lagtime))
        mlags = np.copy(self.mlags)

        if np.any(mlags > maxmlag):
            mlags = mlags[np.where(mlags <= maxmlag)]
            warnings.warn('Dropped lag times exceeding data lengths')

        return mlags

    def compute_models(self, data, n_jobs=None, progress=None, estimate_model_for_lag=None):
        assert estimate_model_for_lag is not None
        n_jobs = handle_n_jobs(n_jobs)
        progress = handle_progress_bar(progress)
        # set lag times

        mlags = self._effective_mlags(data, self.input_lagtime)
        lags = mlags * self.input_lagtime

        predictions = []
        predictions_samples = []
        estimates = []
        estimates_samples = []
        estimated_models = []

        # do we have zero lag? this must be treated separately
        include0 = mlags[0] == 0
        if include0:
            lags_for_estimation = lags[1:]
        else:
            lags_for_estimation = lags

        # estimate models at multiple of input lag time.
        if n_jobs == 1:
            for lag in progress(lags_for_estimation, total=len(lags_for_estimation), leave=False):
                estimated_models.append(estimate_model_for_lag(self.test_estimator, self._test_model, data, lag))
        else:
            fun = estimate_model_for_lag
            args = [(i, fun, (self.test_estimator, self._test_model, data, lag))
                    for i, lag in enumerate(lags_for_estimation)]
            estimated_models = [None for _ in range(len(args))]
            with joining(multiprocessing_context().Pool(processes=n_jobs)) as pool:
                for result in progress(pool.imap_unordered(_imap_wrapper, args),
                                       total=len(lags_for_estimation), leave=False):
                    estimated_models[result[0]] = result[1]

        if include0:
            estimated_models = [None] + estimated_models

        for mlag in mlags:
            # make a prediction using the test model
            predictions.append(self._compute_observables(self._test_model, mlag=mlag))
            # compute prediction errors if we can
            if self.has_errors:
                for sample in self._test_model.samples:
                    predictions_samples.append(self._compute_observables(sample, mlag=mlag))

        for model in estimated_models:
            # evaluate the estimate at lagtime*mlag
            estimates.append(self._compute_observables(model, 1))
            if self.has_errors and self.err_est and model is not None:
                for sample in model.samples:
                    estimates_samples.append(self._compute_observables(sample, mlag=1))

        return LaggedModelValidation(
            estimates=estimates, estimates_samples=estimates_samples,
            predictions=predictions, predictions_samples=predictions_samples, lagtimes=lags)

    def fit(self, data, n_jobs=None, progress=None, estimate_model_for_lag=None, **kw):
        self._model = self.compute_models(data, n_jobs=n_jobs, progress=progress,
                                          estimate_model_for_lag=estimate_model_for_lag)
        return self

    def _compute_observables(self, model, mlag):
        """Compute observables for given model

        Parameters
        ----------
        model : Model
            model to compute observable for.

        mlag : int
            if 1, just compute the observable for given model. If not 1, use
            model to predict result at multiple of given model lagtime.

        Returns
        -------
        Y : ndarray
            array with results

        """
        raise NotImplementedError()


class ChapmanKolmogorovTest:

    def __init__(self, lagtimes, predictions, predictions_samples, estimates, estimates_samples, observable):
        self._lagtimes = np.array(lagtimes)
        self._predictions = np.asfarray(predictions)
        self._predictions_samples = predictions_samples
        self._estimates = np.asfarray(estimates)
        self._estimates_samples = estimates_samples
        self._observable = observable

    @property
    def n_components(self):
        return len(self.estimates[0]) if self.estimates is not None and len(self.estimates) > 0 else 0

    @property
    def lagtimes(self):
        r""" Lagtimes at which estimations and predictions were performed. """
        return self._lagtimes

    @property
    def estimates(self):
        """ Estimates at different lagtimes.

        :getter: each row contains the n observables computed at one of the T lagtimes.
        :type: ndarray(T, n, n)
        """
        return self._estimates

    @property
    def estimates_samples(self):
        r""" Returns the sampled estimates, i.e., a list of arrays as described in :attr:`estimates`. Can be `None`. """
        return self._estimates_samples

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
    def predictions_samples(self):
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
        return self._predictions_samples

    @property
    def has_errors(self):
        return self.predictions_samples is not None and len(self.predictions_samples) > 0

    @property
    def err_est(self):
        return self.estimates_samples is not None

    @staticmethod
    def from_models(models, observable: Observable, test_model=None, include_lag0=True, err_est=False, progress=None):
        assert all(hasattr(m, 'lagtime') for m in models) and (test_model is None or hasattr(test_model, 'lagtime')), \
            "All models and the test model need to have a lagtime property."
        progress = handle_progress_bar(progress)
        models = sorted(models, key=lambda x: x.lagtime)
        if test_model is None:
            test_model = models[0]

        is_bayesian = isinstance(test_model, BayesianModel)
        assert all(isinstance(m, BayesianModel) == is_bayesian for m in models), \
            "All models must either have samples or none at all."
        lagtimes = [0] if include_lag0 else []
        lagtimes += [m.lagtime for m in models]

        predictions = []
        predictions_samples = []
        reference_lagtime = test_model.lagtime
        for lagtime in progress(lagtimes):
            predictions.append(observable(test_model, mlag=lagtime / reference_lagtime))
            if is_bayesian:
                obs = []
                for sample in test_model.samples:
                    obs.append(observable(sample, mlag=lagtime / reference_lagtime))
                predictions_samples.append(obs)

        if include_lag0:
            models = [None] + models
        estimates = []
        estimates_samples = []
        for model in progress(models):
            estimates.append(observable(model, mlag=1))
            if is_bayesian and err_est:
                if model is not None:
                    obs = []
                    for sample in model.samples:
                        obs.append(observable(sample, mlag=1))
                    else:
                        obs.append(observable(model, mlag=0))
                    estimates_samples.append(obs)
                else:
                    estimates_samples.append([observable(model, mlag=0)])

        return ChapmanKolmogorovTest(lagtimes, predictions, predictions_samples, estimates, estimates_samples,
                                     observable)
