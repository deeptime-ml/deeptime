
import math
import numpy as np

from pyemma.util.statistics import confidence_interval
from pyemma.util import types

from sktime.base import Estimator

__author__ = 'noe'


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
        mlags=range(10). The setting None will choose mlags automatically
        according to the longest available trajectory
        Note that you need to be able to do a model prediction for each
        of these lag time multiples, e.g. the value 0 only make sense
        if _predict_observables(0) will work.

    conf : float, default = 0.95
        confidence interval for errors

    err_est : bool, default=False
        if the Estimator is capable of error calculation, will compute
        errors for each tau estimate. This option can be computationally
        expensive.

    n_jobs : int, default=None
        how many jobs to use during calculation

    show_progress : bool, default=True
        Show progressbars for calculation?

    """

    def __init__(self, test_model, test_estimator, mlags=None, conf=0.95, err_est=False,
                 n_jobs=None, show_progress=True):
        super(LaggedModelValidator, self).__init__()
        # set model and estimator
        # copy the test model, since the estimation of cktest modifies the model.
        from copy import deepcopy
        self.test_model = deepcopy(test_model)
        self.test_estimator = test_estimator

        # set mlags
        try:
            maxlength = np.max([len(dtraj) for dtraj in test_estimator.discrete_trajectories_full])
        except AttributeError:
            maxlength = np.max(test_estimator.trajectory_lengths())
        maxmlag = int(math.floor(maxlength / test_estimator.lag))
        if mlags is None:
            mlags = maxmlag
        if types.is_int(mlags):
            mlags = np.arange(mlags)
        mlags = types.ensure_ndarray(mlags, ndim=1, kind='i')
        if np.any(mlags > maxmlag):
            mlags = mlags[np.where(mlags <= maxmlag)]
            self.logger.warning('Changed mlags as some mlags exceeded maximum trajectory length.')
        if np.any(mlags < 0):
            mlags = mlags[np.where(mlags >= 0)]
            self.logger.warning('Changed mlags as some mlags were negative.')
        self.mlags = mlags

        # set conf and error handling
        self.conf = conf
        self.has_errors = issubclass(self.test_model.__class__, SampledModel)
        if self.has_errors:
            self.test_model.set_model_params(conf=conf)
        self.err_est = err_est
        if err_est and not self.has_errors:
            raise ValueError('Requested errors on the estimated models, '
                             'but the model is not able to calculate errors at all')
        self.n_jobs = n_jobs
        self.show_progress = show_progress

    def fit(self, data):
        # lag times
        self._lags = np.array(self.mlags) * self.test_estimator.lag
        pargrid = list(param_grid({'lag': self._lags}))
        # do we have zero lag? this must be treated separately
        include0 = self.mlags[0] == 0
        if include0:
            pargrid = pargrid[1:]

        self._pred = []
        self._pred_L = []
        self._pred_R = []

        self._est = []
        self._est_L = []
        self._est_R = []

        # clone estimators and run estimates

        if self.show_progress:
            if isinstance(self.test_estimator, SampledModel):
                self.test_estimator.show_progress = False
            progress_reporter = self
        else:
            progress_reporter = None

        estimated_models, estimators = \
            estimate_param_scan(self.test_estimator, data, pargrid, return_estimators=True, failfast=False,
                                progress_reporter=progress_reporter, n_jobs=self.n_jobs)
        if include0:
            estimated_models = [None] + estimated_models
            estimators = [None] + estimators

        for i, mlag in enumerate(self.mlags):
            # make a prediction using the current model
            self._pred.append(self._compute_observables(self.test_model, self.test_estimator, mlag))
            # compute prediction errors if we can
            if self.has_errors:
                l, r = self._compute_observables_conf(self.test_model, self.test_estimator, mlag)
                self._pred_L.append(l)
                self._pred_R.append(r)

            # do an estimate at this lagtime
            model = estimated_models[i]
            estimator = estimators[i]
            self._est.append(self._compute_observables(model, estimator))
            if self.has_errors and self.err_est:
                l, r = self._compute_observables_conf(model, estimator)
                self._est_L.append(l)
                self._est_R.append(r)

        # build arrays
        self._est = np.array(self._est)
        self._pred = np.array(self._pred)
        if self.has_errors:
            self._pred_L = np.array(self._pred_L)
            self._pred_R = np.array(self._pred_R)
        else:
            self._pred_L = None
            self._pred_R = None
        if self.has_errors and self.err_est:
            self._est_L = np.array(self._est_L)
            self._est_R = np.array(self._est_R)
        else:
            self._est_L = None
            self._est_R = None

        return self

    @property
    def lagtimes(self):
        return self._lags

    @property
    def estimates(self):
        """ Returns estimates at different lagtimes

        Returns
        -------
        Y : ndarray(T, n)
            each row contains the n observables computed at one of the T lag times.

        """
        return self._est

    @property
    def estimates_conf(self):
        """ Returns the confidence intervals of the estimates at different
        lagtimes (if available).

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
        return self._est_L, self._est_R

    @property
    def predictions(self):
        """ Returns tested model predictions at different lagtimes

        Returns
        -------
        Y : ndarray(T, n)
            each row contains the n observables predicted at one of the T lag
            times by the tested model.

        """
        return self._pred

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
        return self._pred_L, self._pred_R

    # USER functions
    def _compute_observables(self, model, estimator, mlag=1):
        """Compute observables for given model

        Parameters
        ----------
        model : Model
            model to compute observable for.

        estimator : Estimator
            estimator that has produced the model.

        mlag : int, default=1
            if 1, just compute the observable for given model. If not 1, use
            model to predict result at multiple of given model lagtime. Note
            that mlag=0 (no propagation) can occur and should be handled.

        Returns
        -------
        Y : ndarray
            array with results

        """
        raise NotImplementedError('_compute_observables is not implemented. Must override it in subclass!')

    def _compute_observables_conf(self, model, estimator, mlag=1):
        """Compute confidence interval for observables for given model

        Parameters
        ----------
        model : Model
            model to compute observable for. model can be None if mlag=0.
            This scenario must be handled.

        estimator : Estimator
            estimator that has produced the model. estimator can be None if
            mlag=0. This scenario must be handled.

        mlag : int, default=1
            if 1, just compute the observable for given model. If not 1, use
            model to predict result at multiple of given model lagtime. Note
            that mlag=0 (no propagation) can occur and should be handled.

        Returns
        -------
        L : ndarray
            array with lower confidence bounds
        R : ndarray
            array with upper confidence bounds

        """
        raise NotImplementedError('_compute_observables is not implemented. Must override it in subclass!')


class EigenvalueDecayValidator(LaggedModelValidator):

    __serialize_version = 0

    def __init__(self, test_model, estimator, nits=1, mlags=None, conf=0.95,
                 exclude_stat=True, err_est=False, show_progress=True):
        LaggedModelValidator.__init__(self, test_model, estimator, mlags=mlags,
                                      conf=conf, show_progress=show_progress)
        self.nits = nits
        self.exclude_stat = exclude_stat
        self.err_est = err_est  # TODO: this is currently unused

    def _compute_observables(self, model, estimator, mlag=1):
        # for lag time 0 we return all 1's.
        if mlag == 0 or model is None:
            return np.ones(self.nits+1)
        # otherwise compute or predict them from them model
        Y = model.eigenvalues(self.nits+1)
        if self.exclude_stat:
            Y = Y[1:]
        if mlag != 1:
            Y = np.power(Y, mlag)
        return Y

    def _compute_observables_conf(self, model, estimator, mlag=1):
        # for lag time 0 we return all 1's.
        if mlag == 0 or model is None:
            return np.ones(self.nits+1), np.ones(self.nits+1)
        # otherwise compute or predict them from them model
        samples = self.test_model.sample_f('eigenvalues', self.nits+1)
        if mlag != 1:
            for i in range(len(samples)):
                samples[i] = np.power(samples[i], mlag)
        l, r = confidence_interval(samples, conf=self.conf)
        if self.exclude_stat:
            l = l[1:]
            r = r[1:]
        return l, r


class ChapmanKolmogorovValidator(LaggedModelValidator):
    r""" Validates a model estimated at lag time tau by testing its predictions
    for longer lag times

    Parameters
    ----------
    test_model : Model
        Model to be tested

    test_estimator : Estimator
        Parametrized Estimator that has produced the model

    memberships : ndarray(n, m)
        Set memberships to calculate set probabilities. n must be equal to
        the number of active states in model. m is the number of sets.
        memberships must be a row-stochastic matrix (the rows must sum up
        to 1).

    mlags : int or int-array, default=10
        multiples of lag times for testing the Model, e.g. range(10).
        A single int will trigger a range, i.e. mlags=10 maps to
        mlags=range(10). The setting None will choose mlags automatically
        according to the longest available trajectory
        Note that you need to be able to do a model prediction for each
        of these lag time multiples, e.g. the value 0 only make sense
        if _predict_observables(0) will work.

    conf : float, default = 0.95
        confidence interval for errors

    err_est : bool, default=False
        if the Estimator is capable of error calculation, will compute
        errors for each tau estimate. This option can be computationally
        expensive.

    n_jobs : int, default=None
        how many jobs to use during calculation

    show_progress : bool, default=True
        Show progressbars for calculation?
    """
    def __init__(self, test_model, test_estimator, memberships, mlags=None, conf=0.95,
                 err_est=False, n_jobs=None, show_progress=True):
        self.memberships = memberships
        LaggedModelValidator.__init__(self, test_model, test_estimator, mlags=mlags,
                                      conf=conf, n_jobs=n_jobs,
                                      show_progress=show_progress)
        self.err_est = err_est  # TODO: this is currently unused

    @property
    def memberships(self):
        return self._memberships

    @memberships.setter
    def memberships(self, value):
        self._memberships = types.ensure_ndarray(value, ndim=2, kind='numeric')
        self.nstates, self.nsets = self._memberships.shape
        assert np.allclose(self._memberships.sum(axis=1), np.ones(self.nstates))  # stochastic matrix?

    # TODO: do not store this, obtain active set from model during fit!
    @property
    def test_estimator(self):
        return self._test_estimator

    @test_estimator.setter
    def test_estimator(self, test_estimator):
        self._test_estimator = test_estimator
        self.active_set = types.ensure_ndarray(np.array(test_estimator.active_set), kind='i')  # create a copy
        # map from the full set (here defined by the largest state index in active set) to active
        self._full2active = np.zeros(np.max(self.active_set)+1, dtype=int)
        self._full2active[self.active_set] = np.arange(self.nstates)

    @property
    def test_model(self):
        return self._test_model

    @test_model.setter
    def test_model(self, test_model):
        self._test_model = test_model
        # define starting distribution
        self.P0 = self.memberships * test_model.stationary_distribution[:, None]
        self.P0 /= self.P0.sum(axis=0)  # column-normalize

    def _compute_observables(self, model, estimator, mlag=1):
        # for lag time 0 we return an identity matrix
        if mlag == 0 or model is None:
            return np.eye(self.nsets)
        # otherwise compute or predict them by model.propagate
        pk_on_set = np.zeros((self.nsets, self.nsets))
        subset = self._full2active[model.active_set]  # find subset we are now working on
        for i in range(self.nsets):
            p0 = self.P0[:, i]  # starting distribution on reference active set
            p0sub = p0[subset]  # map distribution to new active set
            p0sub /= p0sub.sum()  # renormalize
            pksub = model.propagate(p0sub, mlag)
            for j in range(self.nsets):
                pk_on_set[i, j] = np.dot(pksub, self.memberships[subset, j])  # map onto set
        return pk_on_set

    def _compute_observables_conf(self, model, estimator, mlag=1):
        # for lag time 0 we return an identity matrix
        assert model is not estimator
        if mlag == 0 or model is None:
            return np.eye(self.nsets), np.eye(self.nsets)
        # otherwise compute or predict them by model.propagate
        subset = self._full2active[estimator.active_set]  # find subset we are now working on
        l = np.zeros((self.nsets, self.nsets))
        r = np.zeros((self.nsets, self.nsets))
        for i in range(self.nsets):
            p0 = self.P0[:, i]  # starting distribution
            p0sub = p0[subset]  # map distribution to new active set
            p0sub /= p0sub.sum()  # renormalize
            pksub_samples = model.sample_f('propagate', p0sub, mlag)
            for j in range(self.nsets):
                pk_on_set_samples = np.fromiter((np.dot(pksub, self.memberships[subset, j])
                                                 for pksub in pksub_samples), dtype=np.float, count=len(pksub_samples))
                l[i, j], r[i, j] = confidence_interval(pk_on_set_samples, conf=self.conf)
        return l, r

# TODO: conf is better added to function sample_conf() and not made a model parameter
# TODO: should Estimator really have a model parameter? This is not consistent with sklearn
# TODO: estimate_param_scan without return_estimators=True doesn't work at all!
