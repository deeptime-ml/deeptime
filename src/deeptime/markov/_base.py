import abc

import numpy as np

from ..base import Estimator, BayesianModel
from ._observables import MembershipsObservable


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

    def ck_test(self, models, n_metastable_sets, include_lag0=True, err_est=False, progress=None):
        r""" Performs a Chapman Kolmogorov test.
        See :meth:`MarkovStateModel.ck_test <deeptime.markov.msm.MarkovStateModel.ck_test>` for more details """
        clustering = self.prior.pcca(n_metastable_sets)
        observable = MembershipsObservable(self, clustering, initial_distribution=self.prior.stationary_distribution)
        from deeptime.util.validation import ck_test
        return ck_test(models, observable, test_model=self, include_lag0=include_lag0,
                       err_est=err_est, progress=progress)
