import abc
from typing import Optional, Callable

import numpy as np

from ..base import Estimator, Model
from ..util.types import ensure_dtraj_list


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
        statistics : QuantityStatistics
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
