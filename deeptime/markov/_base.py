import abc

import numpy as np

from ..util import confidence_interval, LaggedModelValidator
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


class BayesianPosterior(Model):
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
        r"""
        The sampled models

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
        submodel : BayesianPosterior
            A posterior with prior and samples restricted to specified states.
        """
        return BayesianPosterior(
            prior=self.prior.submodel(states),
            samples=[sample.submodel(states) for sample in self.samples]
        )

    def gather_stats(self, quantity, store_samples=False, delimiter='/', confidence=0.95, *args, **kwargs):
        """ Obtain statistics about a sampled quantity. Can also be a chained call, separated by the delimiter.

        Parameters
        ----------
        quantity: str
            name of attribute, which will be evaluated on samples
        store_samples: bool, optional, default=False
            whether to store the samples (array).
        delimiter : str, optional, default='/'
            separator to call members of members
        confidence : float, optional, default=0.95
            Size of the confidence intervals.
        *args
            pass-through
        **kwargs
            pass-through

        Returns
        -------
        statistics : deeptime.util.QuantityStatistics
            The statistics
        """
        from deeptime.util import QuantityStatistics
        return QuantityStatistics.gather(self.samples, quantity=quantity, store_samples=store_samples,
                                         delimiter=delimiter, confidence=confidence, *args, **kwargs)

    def evaluate_samples(self, quantity, delimiter='/', *args, **kwargs):
        r""" Obtains a quantity (like an attribute or result of a method or a property) from each of the samples.
        Returns as list.

        Parameters
        ----------
        quantity : str
            The quantity. Can be also deeper in the instance hierarchy, indicated by the delimiter.
        delimiter : str, default='/'
            The delimiter.
        *args
            Arguments passed to the evaluation point of the quantity.
        **kwargs
            Keyword arguments passed to the evaluation point of the quantity.

        Returns
        -------
        result : list of any or ndarray
            A list of the quantity evaluated on each of the samples. If can be converted to float ndarray then ndarray.
        """
        from deeptime.util.stats import evaluate_samples as _eval
        return _eval(self.samples, quantity=quantity, delimiter=delimiter, *args, **kwargs)


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
    conf : float, default = 0.95
        confidence interval for errors

    Notes
    -----
    This is an adaption of the Chapman-Kolmogorov Test described in detail
    in :footcite:`prinz2011markov` to Hidden MSMs as described in :footcite:`noe2013projected`.

    References
    ----------
    .. footbibliography::
    """

    def __init__(self, test_model, test_estimator, memberships, test_model_lagtime,
                 mlags, conf=0.95):
        super().__init__(test_model, test_estimator, test_model_lagtime=test_model_lagtime, conf=conf, mlags=mlags)
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

    def _compute_observables_conf(self, model, mlag, conf=0.95):
        # otherwise compute or predict them by model.propagate
        if mlag == 0 or model is None:
            return np.eye(self.nsets), np.eye(self.nsets)
        prior = model.prior
        if hasattr(prior, 'transition_model'):
            symbols = prior.transition_model.count_model.state_symbols
        else:
            symbols = prior.count_model.state_symbols
        subset = self._full2active[symbols]  # find subset we are now working on
        l = np.zeros((self.nsets, self.nsets))
        r = np.zeros((self.nsets, self.nsets))
        for i in range(self.nsets):
            p0 = self.P0[:, i]  # starting distribution
            p0sub = p0[subset]  # map distribution to new active set
            p0sub /= p0sub.sum()  # renormalize
            pksub_samples = []
            for m in model.samples:
                if hasattr(m, 'transition_model'):
                    m = m.transition_model
                pksub_samples.append(m.propagate(p0sub, mlag))
            for j in range(self.nsets):
                pk_on_set_samples = np.fromiter((np.dot(pksub, self.memberships[subset, j])
                                                 for pksub in pksub_samples), dtype=np.float32,
                                                count=len(pksub_samples))
                l[i, j], r[i, j] = confidence_interval(pk_on_set_samples, conf=self.conf)
        return l, r
