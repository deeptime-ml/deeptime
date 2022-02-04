import abc

import numpy as np

from ..util import LaggedModelValidator, LaggedModelValidation
from ..util.types import ensure_array
from ..base import Estimator, Model


class _MSMBaseEstimator(Estimator, metaclass=abc.ABCMeta):
    r"""Maximum likelihood estimator for MSMs given discrete trajectory statistics.
    """

    def __init__(self, reversible=True, sparse=False):
        r""" Creates a new base estimator instance.

        Parameters
        ----------
        reversible : bool, optional, default = True
            If true compute reversible MarkovStateModel, else non-reversible MarkovStateModel
        sparse : bool, optional, default = False
            If true compute count matrix, transition matrix and all derived
            quantities using sparse matrix algebra. In this case python sparse
            matrices will be returned by the corresponding functions instead of
            numpy arrays. This behavior is suggested for very large numbers of
            states (e.g. > 4000) because it is likely to be much more efficient.
        """
        super(_MSMBaseEstimator, self).__init__()
        self.reversible = reversible
        self.sparse = sparse

    @property
    def reversible(self) -> bool:
        r""" If true compute reversible MarkovStateModel, else non-reversible MarkovStateModel """
        return self._reversible

    @reversible.setter
    def reversible(self, value: bool):
        self._reversible = value

    @property
    def sparse(self) -> bool:
        r"""  If true compute count matrix, transition matrix and all derived quantities using sparse matrix algebra.
        In this case python sparse matrices will be returned by the corresponding functions instead of numpy arrays.
        This behavior is suggested for very large numbers of states (e.g. > 4000) because it is likely
        to be much more efficient.
        """
        return self._sparse

    @sparse.setter
    def sparse(self, value: bool):
        self._sparse = value


class BayesianMSMPosterior(BayesianModel):
    r""" Bayesian posterior from bayesian MSM sampling.

    Parameters
    ----------
    prior : deeptime.markov.msm.MarkovStateModel, optional, default=None
        The prior.
    samples : list of deeptime.markov.msm.MarkovStateModel, optional, default=None
        Sampled models.

    See Also
    --------
    deeptime.markov.msm.BayesianMSM
    """

    def __init__(self, prior=None, samples=None):
        super().__init__()
        self._prior = prior
        self._samples = samples

    @property
    def samples(self):
        r""" The sampled models

        Returns
        -------
        models : list of deeptime.markov.msm.MarkovStateModel or None
            samples
        """
        return self._samples

    @property
    def prior(self):
        r"""
        The prior model.

        Returns
        -------
        prior : deeptime.markov.msm.MarkovStateModel or None
            the prior
        """
        return self._prior

    def __iter__(self):
        for s in self.samples:
            yield s

    def submodel(self, states: np.ndarray):
        r""" Creates a bayesian posterior that is restricted onto the specified states.

        Parameters
        ----------
        states: (N,) ndarray, dtype=int
            array of integers specifying the states to restrict to

        Returns
        -------
        submodel : BayesianMSMPosterior
            A posterior with prior and samples restricted to specified states.
        """
        return BayesianMSMPosterior(
            prior=self.prior.submodel(states),
            samples=[sample.submodel(states) for sample in self.samples]
        )

    def timescales(self, k=None):
        r""" Relaxation timescales corresponding to the eigenvalues.

        Parameters
        ----------
        k : int, optional, default=None
            The number of timescales (excluding the stationary process).

        Returns
        -------
        timescales : tuple(iterable, iterable)
            Timescales of the prior and timescales of the samples.
        """
        return self.prior.timescales(k=k), self.evaluate_samples('timescales', k=k)

    @property
    def lagtime(self):
        r"""Lagtime of the models."""
        return self.prior.lagtime


class MembershipsChapmanKolmogorovValidator(LaggedModelValidator):
    r""" Validates a model estimated at lag time tau by testing its predictions for longer lag times.
    This is known as the Chapman-Kolmogorov test as it is based on the Chapman-Kolmogorov equation.

    The test is performed on metastable sets of states rather than the microstates themselves.

    Parameters
    ----------
    memberships : ndarray(n, m)
        Set memberships to calculate set probabilities. n must be equal to
        the number of active states in model. m is the number of sets.
        memberships must be a row-stochastic matrix (the rows must sum up to 1).
    test_model : Model
        Model to be tested
    test_estimator : Estimator
        Parametrized Estimator that has produced the model
    mlags : int or int-array
        Multiples of lag times for testing the Model, e.g. range(10).
        A single int will trigger a range, i.e. `mlags=10` maps to
        `mlags=range(10)`. The setting None will choose mlags automatically
        according to the longest available trajectory.
        Note that you need to be able to do a model prediction for each
        of these lag time multiples, e.g. the value 0 only make sense
        if _predict_observables(0) will work.

    Notes
    -----
    This is an adaption of the Chapman-Kolmogorov Test described in detail
    in :footcite:`prinz2011markov` to Hidden MSMs as described in :footcite:`noe2013projected`.

    References
    ----------
    .. footbibliography::
    """

    def __init__(self, test_model, test_estimator, memberships, test_model_lagtime, mlags):
        super().__init__(test_model, test_estimator, test_model_lagtime=test_model_lagtime, mlags=mlags)
        self.memberships = memberships
        self.test_model = test_model

    @property
    def test_model(self):
        return self._test_model

    @test_model.setter
    def test_model(self, test_model):
        assert self.memberships is not None
        if hasattr(test_model, 'prior'):
            m = test_model.prior
        else:
            m = test_model
        if hasattr(m, 'transition_model'):
            m = m.transition_model
        n_states = m.n_states
        statdist = m.stationary_distribution
        symbols = m.count_model.state_symbols
        symbols_full = m.count_model.n_states_full
        assert self.memberships.shape[0] == n_states, 'provided memberships and test_model n_states mismatch'
        self._test_model = test_model
        # define starting distribution
        P0 = self.memberships * statdist[:, None]
        P0 /= P0.sum(axis=0)  # column-normalize
        self.P0 = P0

        # map from the full set (here defined by the largest state index in active set) to active
        self._full2active = np.zeros(np.max(symbols_full) + 1, dtype=int)
        self._full2active[symbols] = np.arange(len(symbols))

    @property
    def memberships(self):
        return self._memberships

    @memberships.setter
    def memberships(self, value):
        self._memberships = ensure_array(value, ndim=2, dtype=float)
        self.nstates, self.nsets = self._memberships.shape
        assert np.allclose(self._memberships.sum(axis=1), np.ones(self.nstates))  # stochastic matrix?

    def _compute_observables(self, model, mlag):
        if mlag == 0 or model is None:
            return np.eye(self.nsets)
        # otherwise compute or predict them by model.propagate
        pk_on_set = np.zeros((self.nsets, self.nsets))
        # compute observable on prior in case for Bayesian models.
        if hasattr(model, 'prior'):
            model = model.prior
        if hasattr(model, 'transition_model'):
            model = model.transition_model
        symbols = model.count_model.state_symbols
        subset = self._full2active[symbols]  # find subset we are now working on
        for i in range(self.nsets):
            p0 = self.P0[:, i]  # starting distribution on reference active set
            p0sub = p0[subset]  # map distribution to new active set
            if subset is not None:
                p0sub /= p0sub.sum()  # renormalize
            pksub = model.propagate(p0sub, mlag)
            for j in range(self.nsets):
                pk_on_set[i, j] = np.dot(pksub, self.memberships[subset, j])  # map onto set
        return pk_on_set

    def fit(self, data, n_jobs=None, progress=None, estimate_model_for_lag=None, **kw):
        models = self.compute_models(data, n_jobs, progress, estimate_model_for_lag)
        self._model = MembershipsLaggedModelValidation(models, self.memberships)
        return super().fit(data, n_jobs, progress, estimate_model_for_lag, **kw)


class MembershipsLaggedModelValidation(LaggedModelValidation):
    def __init__(self, models: LaggedModelValidation, memberships: np.ndarray):
        super().__init__(estimates=models.estimates, estimates_samples=models.estimates_samples,
                         predictions=models.predictions, predictions_samples=models.predictions_samples,
                         lagtimes=models.lagtimes)
        self._memberships = memberships

    @property
    def memberships(self):
        return self._memberships

    @property
    def n_states(self):
        return self.memberships.shape[0]

    @property
    def n_sets(self):
        return self.memberships.shape[1]
