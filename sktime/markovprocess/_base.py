import typing

import numpy as np

from sktime.base import Estimator, Model
from sktime.markovprocess import MarkovStateModel
# TODO: we do not need this anymore!
from sktime.util import confidence_interval, ensure_dtraj_list



# TODO: this could me moved to msmtools.dtraj
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
        Lag time at which counting will be done. If sh
    sliding : bool
        True for splitting trajectories for sliding count, False if lag-sampling will be applied
    shift : None or int
        Start of first full tau-window. If None, shift will be randomly generated

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


# TODO: this could me moved to msmtools.dtraj
def cvsplit_dtrajs(dtrajs, random_state=None):
    """ Splits the trajectories into a training and test set with approximately equal number of trajectories

    Parameters
    ----------
    dtrajs : list of ndarray(int)
        Discrete trajectories

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

class _MSMBaseEstimator(Estimator):
    r"""Maximum likelihood estimator for MSMs given discrete trajectory statistics

    Parameters
    ----------
    lag : int
        lag time at which transitions are counted and the transition matrix is
        estimated.

    reversible : bool, optional, default = True
        If true compute reversible MarkovStateModel, else non-reversible MarkovStateModel

    count_mode : str, optional, default='sliding'
        mode to obtain count matrices from discrete trajectories. Should be
        one of:

        * 'sliding' : A trajectory of length T will have :math:`T-\tau` counts
          at time indexes

          .. math::

             (0 \rightarrow \tau), (1 \rightarrow \tau+1), ..., (T-\tau-1 \rightarrow T-1)

        * 'effective' : Uses an estimate of the transition counts that are
          statistically uncorrelated. Recommended when used with a
          Bayesian MarkovStateModel.
        * 'sample' : A trajectory of length T will have :math:`T/\tau` counts
          at time indexes

          .. math::

                (0 \rightarrow \tau), (\tau \rightarrow 2 \tau), ..., (((T/\tau)-1) \tau \rightarrow T)

    sparse : bool, optional, default = False
        If true compute count matrix, transition matrix and all derived
        quantities using sparse matrix algebra. In this case python sparse
        matrices will be returned by the corresponding functions instead of
        numpy arrays. This behavior is suggested for very large numbers of
        states (e.g. > 4000) because it is likely to be much more efficient.

    dt_traj : str, optional, default='1 step'
        Description of the physical time of the input trajectories. May be used
        by analysis algorithms such as plotting tools to pretty-print the axes.
        By default '1 step', i.e. there is no physical time unit. Specify by a
        number, whitespace and unit. Permitted units are (* is an arbitrary
        string). E.g. 200 picoseconds or 200ps.

    mincount_connectivity : float or '1/n'
        minimum number of counts to consider a connection between two states.
        Counts lower than that will count zero in the connectivity check and
        may thus separate the resulting transition matrix. The default
        evaluates to 1/n_states.

    """

    def __init__(self, lagtime=1, reversible=True, count_mode='sliding', sparse=False,
                 dt_traj='1 step', mincount_connectivity='1/n'):
        super(_MSMBaseEstimator, self).__init__()
        self.lagtime = lagtime

        # set basic parameters
        self.reversible = reversible

        # sparse matrix computation wanted?
        self.sparse = sparse

        # store counting mode (lowercase)
        self.count_mode = count_mode
        if self.count_mode not in ('sliding', 'effective', 'sample'):
            raise ValueError('count mode ' + count_mode + ' is unknown.')

        # time step
        self.dt_traj = dt_traj

        # connectivity
        self.mincount_connectivity = mincount_connectivity


class BayesianPosterior(Model):
    def __init__(self,
                 prior: typing.Optional[MarkovStateModel] = None,
                 samples: typing.Optional[typing.List[MarkovStateModel]] = None):
        self.prior = prior
        self.samples = samples

    def __iter__(self):
        for s in self.samples:
            yield s

    def gather_stats(self, quantity, store_samples=False, *args, **kwargs):
        """ obtain statistics about a sampled quantity

        Parameters
        ----------
        quantity: str
            name of attribute, which will be evaluated on samples

        store_samples: bool, optional, default=False
            whether to store the samples (array).
        *args:

        """
        from sktime.util import call_member
        samples = [call_member(s, quantity, *args, **kwargs) for s in self]
        return QuantityStatistics(samples, quantity=quantity, store_samples=store_samples)

    def submodel_largest(self, strong=True, mincount_connectivity='1/n', observe_nonempty=True, dtrajs=None):
        dtrajs = ensure_dtraj_list(dtrajs)
        states = self.prior.states_largest(strong=strong, mincount_connectivity=mincount_connectivity)
        obs = self.prior.nonempty_obs(dtrajs) if observe_nonempty else None
        return self.submodel(states=states, obs=obs, mincount_connectivity=mincount_connectivity)

    def submodel_populous(self, strong=True, mincount_connectivity='1/n', observe_nonempty=True, dtrajs=None):
        dtrajs = ensure_dtraj_list(dtrajs)
        states = self.prior.states_populous(strong=strong, mincount_connectivity=mincount_connectivity)
        obs = self.prior.nonempty_obs(dtrajs) if observe_nonempty else None
        return self.submodel(states=states, obs=obs, mincount_connectivity=mincount_connectivity)

    def submodel(self, states=None, obs=None, mincount_connectivity='1/n'):
        # restrict prior
        sub_model = self.prior.submodel(states=states, obs=obs,
                                        mincount_connectivity=mincount_connectivity)
        # restrict reduce samples
        count_model = sub_model.count_model
        subsamples = [sample.submodel(states=count_model.active_set, obs=count_model.observable_set)
                      for sample in self]
        return BayesianPosterior(sub_model, subsamples)


class QuantityStatistics(Model):
    """ Container for statistical quantities computed on samples.

    Parameters
    ----------
    samples: list of ndarrays
        the samples
    store_samples: bool, default=False
        whether to store the samples (array).

    Attributes
    ----------
    mean: array(n)
        mean along axis=0
    std: array(n)
        std deviation along axis=0
    L : ndarray(shape)
        element-wise lower bounds
    R : ndarray(shape)
        element-wise upper bounds

    """

    def __init__(self, samples: typing.List[np.ndarray], quantity, store_samples=False):
        self.quantity = quantity
        # TODO: shall we refer to the original object?
        # we re-add the (optional) quantity, because the creation of a new array will strip it.
        unit = getattr(samples[0], 'u', None)
        if unit is not None:
            samples = np.array(tuple(x.magnitude for x in samples))
        else:
            samples = np.array(samples)
        if unit is not None:
            samples *= unit
        if store_samples:
            self.samples = samples
        else:
            self.samples = np.empty(0) * unit
        self.mean = samples.mean(axis=0)
        self.std = samples.std(axis=0)
        self.L, self.R = confidence_interval(samples)
        if unit is not None:
            self.L *= unit
            self.R *= unit


def score_cv(estimator: _MSMBaseEstimator, dtrajs, n=10, score_method='VAMP2', score_k=10, random_state=None):
    r""" Scores the MSM using the variational approach for Markov processes [1]_ [2]_ and cross-validation [3]_ .

    Divides the data into training and test data, fits a MSM using the training
    data using the parameters of this estimator, and scores is using the test
    data.
    Currently only one way of splitting is implemented, where for each n,
    the data is randomly divided into two approximately equally large sets of
    discrete trajectory fragments with lengths of at least the lagtime.

    Currently only implemented using dense matrices - will be slow for large state spaces.

    Parameters
    ----------
    estimator : MSMBaseEstimator like
        estimator to produce models for CV.
    dtrajs : list of arrays
        Test data (discrete trajectories).
    n : number of samples
        Number of repetitions of the cross-validation. Use large n to get solid
        means of the score.
    score_method : str, optional, default='VAMP2'
        Overwrite scoring method to be used if desired. If `None`, the estimators scoring
        method will be used.
        Available scores are based on the variational approach for Markov processes [1]_ [2]_ :

        *  'VAMP1'  Sum of singular values of the symmetrized transition matrix [2]_ .
                    If the MSM is reversible, this is equal to the sum of transition
                    matrix eigenvalues, also called Rayleigh quotient [1]_ [3]_ .
        *  'VAMP2'  Sum of squared singular values of the symmetrized transition matrix [2]_ .
                    If the MSM is reversible, this is equal to the kinetic variance [4]_ .

    score_k : int or None
        The maximum number of eigenvalues or singular values used in the
        score. If set to None, all available eigenvalues will be used.

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
    if estimator.count_mode not in ('sliding', 'sample'):
        raise ValueError('score_cv currently only supports count modes "sliding" and "sample"')
    sliding = estimator.count_mode == 'sliding'
    scores = []
    for fold in range(n):
        dtrajs_split = blocksplit_dtrajs(dtrajs, lag=estimator.lagtime, sliding=sliding, random_state=random_state)
        dtrajs_train, dtrajs_test = cvsplit_dtrajs(dtrajs_split, random_state=random_state)
        model = estimator.fit(dtrajs_train).fetch_model()
        s = model.score(dtrajs_test, score_method=score_method, score_k=score_k)
        scores.append(s)
    return np.array(scores)

