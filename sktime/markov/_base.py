import abc
from typing import Optional, Callable

import numpy as np

from ..base import Estimator, Model


def blocksplit_dtrajs(dtrajs, lag=1, sliding=True, shift=None, random_state=None):
    """ Splits the discrete trajectories into approximately uncorrelated fragments

    Will split trajectories into fragments of lengths lag or longer. These fragments
    are overlapping in order to conserve the transition counts at given lag.
    If sliding=True, the resulting trajectories will lead to exactly the same count
    matrix as when counted from dtrajs. If sliding=False (sampling at lag), the
    count matrices are only equal when also setting shift=0.

    Parameters
    ----------
    dtrajs : list of ndarray(int)
        Discrete trajectories
    lag : int
        Lag time at which counting will be done.
    sliding : bool
        True for splitting trajectories for sliding count, False if lag-sampling will be applied
    shift : None or int
        Start of first full tau-window. If None, shift will be randomly generated
    random_state : None or int or np.random.RandomState
        Random seed to use.
    """
    from sklearn.utils.random import check_random_state
    dtrajs_new = []
    random_state = check_random_state(random_state)
    for dtraj in dtrajs:
        if len(dtraj) <= lag:
            continue
        if shift is None:
            s = random_state.randint(min(lag, dtraj.size - lag))
        else:
            s = shift
        if sliding:
            if s > 0:
                dtrajs_new.append(dtraj[0:lag + s])
            for t0 in range(s, dtraj.size - lag, lag):
                dtrajs_new.append(dtraj[t0:t0 + 2 * lag])
        else:
            for t0 in range(s, dtraj.size - lag, lag):
                dtrajs_new.append(dtraj[t0:t0 + lag + 1])
    return dtrajs_new


def cvsplit_dtrajs(dtrajs, random_state=None):
    """ Splits the trajectories into a training and test set with approximately equal number of trajectories

    Parameters
    ----------
    dtrajs : list of ndarray(int)
        Discrete trajectories
    random_state : None or int or np.random.RandomState
        Random seed to use.
    """
    from sklearn.utils.random import check_random_state
    if len(dtrajs) == 1:
        raise ValueError('Only have a single trajectory. Cannot be split into train and test set')
    random_state = check_random_state(random_state)
    I0 = random_state.choice(len(dtrajs), int(len(dtrajs) / 2), replace=False)
    I1 = np.array(list(set(list(np.arange(len(dtrajs)))) - set(list(I0))))
    dtrajs_train = [dtrajs[i] for i in I0]
    dtrajs_test = [dtrajs[i] for i in I1]
    return dtrajs_train, dtrajs_test


class _MSMBaseEstimator(Estimator, metaclass=abc.ABCMeta):
    r"""Maximum likelihood estimator for MSMs given discrete trajectory statistics

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

    def __init__(self, reversible=True, sparse=False):
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

    See Also
    --------
    sktime.markov.msm.BayesianMSM : bayesian posterior estimator
    """
    def __init__(self, prior=None, samples=None):
        r""" Creates a new instance of this type of model.

        Parameters
        ----------
        prior : sktime.markov.msm.MarkovStateModel, optional, default=None
            The prior.
        samples : list of sktime.markov.msm.MarkovStateModel, optional, default=None
            Sampled models.
        """
        super().__init__()
        self._prior = prior
        self._samples = samples

    @property
    def samples(self):
        r"""
        The sampled models

        Returns
        -------
        models : list of sktime.markov.msm.MarkovStateModel or None
            samples
        """
        return self._samples

    @property
    def prior(self):
        r"""
        The prior model.

        Returns
        -------
        prior : sktime.markov.msm.MarkovStateModel or None
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
        from sktime.util import QuantityStatistics
        return QuantityStatistics.gather(self.samples, quantity=quantity, store_samples=store_samples,
                                         delimiter=delimiter, confidence=confidence, *args, **kwargs)


def score_cv(fit_fetch: Callable, dtrajs, lagtime, n=10, count_mode="sliding", score_method='VAMP2',
             score_k: int = 10, blocksplit: bool = True, random_state=None):
    r""" Scores the MSM using the variational approach for Markov processes and cross-validation.

    Implementation and ideas following [1]_ [2]_ and cross-validation [3]_.

    Divides the data into training and test data, fits a MSM using the training
    data using the parameters of this estimator, and scores is using the test
    data.
    Currently only one way of splitting is implemented, where for each n,
    the data is randomly divided into two approximately equally large sets of
    discrete trajectory fragments with lengths of at least the lagtime.

    Currently only implemented using dense matrices - will be slow for large state spaces.

    Parameters
    ----------
    fit_fetch : callable, optional, default=None
        Can be provided for a custom fit and fetch method. Should be a function pointer or lambda which
        takes a list of discrete trajectories as input and yields an estimated MSM or MSM-like model.
    dtrajs : list of array_like
        Test data (discrete trajectories).
    lagtime : int
        lag time
    n : number of samples
        Number of repetitions of the cross-validation. Use large n to get solid
        means of the score.
    count_mode : str, optional, default='sliding'
        counting mode of count matrix estimator, if sliding the trajectory is split in a sliding window fashion.
        Supports 'sliding' and 'sample'.
    score_method : str, optional, default='VAMP2'
        Overwrite scoring method to be used if desired. If `None`, the estimators scoring
        method will be used.
        Available scores are based on the variational approach for Markov processes [1]_ [2]_ :

        *  'VAMP1'  Sum of singular values of the symmetrized transition matrix [2]_ .
                    If the MSM is reversible, this is equal to the sum of transition
                    matrix eigenvalues, also called Rayleigh quotient [1]_ [3]_ .
        *  'VAMP2'  Sum of squared singular values of the symmetrized transition matrix [2]_ .
                    If the MSM is reversible, this is equal to the kinetic variance [4]_ .

    blocksplit : bool, optional, default=True
        Whether to perform blocksplitting (see :meth:`blocksplit_dtrajs` ) before evaluating folds. Defaults to `True`.
    score_k : int or None
        The maximum number of eigenvalues or singular values used in the
        score. If set to None, all available eigenvalues will be used.
    random_state : None or int or np.random.RandomState
        Random seed to use.

    References
    ----------
    .. [1] Noe, F. and F. Nueske: A variational approach to modeling slow processes
        in stochastic dynamical systems. SIAM Multiscale Model. Simul. 11, 635-655 (2013).
    .. [2] Wu, H and F. Noe: Variational approach for learning Markov processes
        from time series data (in preparation).
    .. [3] McGibbon, R and V. S. Pande: Variational cross-validation of slow
        dynamical modes in molecular kinetics, J. Chem. Phys. 142, 124105 (2015).
    .. [4] Noe, F. and C. Clementi: Kinetic distance and kinetic maps from molecular
        dynamics simulation. J. Chem. Theory Comput. 11, 5002-5011 (2015).

    """
    from sktime.util import ensure_dtraj_list
    dtrajs = ensure_dtraj_list(dtrajs)  # ensure format
    if count_mode not in ('sliding', 'sample'):
        raise ValueError('score_cv currently only supports count modes "sliding" and "sample"')
    sliding = count_mode == 'sliding'
    scores = []
    for fold in range(n):
        if blocksplit:
            dtrajs_split = blocksplit_dtrajs(dtrajs, lag=lagtime, sliding=sliding, random_state=random_state)
        else:
            dtrajs_split = dtrajs
            if len(dtrajs_split) <= 1:
                raise ValueError("Need at least two trajectories if blocksplit is not used to decompose the data.")
        dtrajs_train, dtrajs_test = cvsplit_dtrajs(dtrajs_split, random_state=random_state)

        model = fit_fetch(dtrajs_train)
        s = model.score(dtrajs_test, score_method=score_method, score_k=score_k)
        scores.append(s)
    return np.array(scores)
