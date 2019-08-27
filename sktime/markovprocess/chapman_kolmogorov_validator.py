
import math
import numpy as np

from pyemma.util.statistics import confidence_interval
from pyemma.util import types

from sktime.base import Estimator, Model
from sktime.lagged_model_validator import LaggedModelValidator
from sktime.markovprocess import MarkovStateModel, BayesianMSMPosterior

__author__ = 'noe, marscher'


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
                 err_est=False):
        self.memberships = memberships
        super(ChapmanKolmogorovValidator, self).__init__(test_model, test_estimator, conf=conf, mlags=mlags)
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
        #TODO: active_set is a model attribute
        self.active_set = types.ensure_ndarray(np.array(test_estimator.active_set), kind='i')  # create a copy
        # map from the full set (here defined by the largest state index in active set) to active
        self._full2active = np.zeros(np.max(self.active_set)+1, dtype=int)
        self._full2active[self.active_set] = np.arange(self.nstates)

    @property
    def test_model(self):
        return self._test_model

    @test_model.setter
    def test_model(self, test_model: MarkovStateModel):
        self._test_model = test_model.copy()
        # define starting distribution
        self.P0 = self.memberships * test_model.stationary_distribution[:, None]
        self.P0 /= self.P0.sum(axis=0)  # column-normalize

    def _compute_observables(self, model: MarkovStateModel, mlag=1):
        # for lag time 0 we return an identity matrix
        if mlag == 0 or model is None:
            return np.eye(self.nsets)
        # otherwise compute or predict them by model.propagate
        pk_on_set = np.zeros((self.nsets, self.nsets))
        if model.count_model is not None:
            subset = self._full2active[model.count_model.active_set]  # find subset we are now working on
        else:
            # TODO: even needed?
            subset = self._full2active[np.arange(model.nstates)]
        for i in range(self.nsets):
            p0 = self.P0[:, i]  # starting distribution on reference active set
            p0sub = p0[subset]  # map distribution to new active set
            p0sub /= p0sub.sum()  # renormalize
            pksub = model.propagate(p0sub, mlag)
            for j in range(self.nsets):
                pk_on_set[i, j] = np.dot(pksub, self.memberships[subset, j])  # map onto set
        return pk_on_set

    def _compute_observables_conf(self, model: BayesianMSMPosterior, mlag=1, conf=0.95):
        # for lag time 0 we return an identity matrix
        if mlag == 0 or model is None:
            return np.eye(self.nsets), np.eye(self.nsets)
        # otherwise compute or predict them by model.propagate
        if model.prior.count_model is not None:
            subset = self._full2active[model.prior.count_model.active_set]  # find subset we are now working on
        else:
            # TODO: even needed?
            subset = self._full2active[np.arange(model.prior.nstates)]
        l = np.zeros((self.nsets, self.nsets))
        r = np.zeros((self.nsets, self.nsets))
        for i in range(self.nsets):
            p0 = self.P0[:, i]  # starting distribution
            p0sub = p0[subset]  # map distribution to new active set
            p0sub /= p0sub.sum()  # renormalize
            pksub_samples = [m.propagate(p0sub, mlag) for m in model.samples]
            for j in range(self.nsets):
                pk_on_set_samples = np.fromiter((np.dot(pksub, self.memberships[subset, j])
                                                 for pksub in pksub_samples), dtype=np.float, count=len(pksub_samples))
                l[i, j], r[i, j] = confidence_interval(pk_on_set_samples, conf=self.conf)
        return l, r
