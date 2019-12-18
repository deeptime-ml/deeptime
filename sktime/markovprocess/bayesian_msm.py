import typing

from sktime.markovprocess._base import _MSMBaseEstimator, BayesianPosterior
from sktime.markovprocess.maximum_likelihood_msm import MaximumLikelihoodMSM
from .markov_state_model import MarkovStateModel

__author__ = 'noe, marscher'


class BayesianMSM(_MSMBaseEstimator):
    r""" Bayesian estimator for MSMs given discrete trajectory statistics

    Parameters
    ----------
    lag : int, optional, default=1
       lagtime to estimate the HMSM at

    nsamples : int, optional, default=100
       number of sampled transition matrices used

    nsteps : int, optional, default=None
       number of Gibbs sampling steps for each transition matrix used.
       If None, nstep will be determined automatically

    reversible : bool, optional, default = True
       If true compute reversible MSM, else non-reversible MSM

    statdist_constraint : (M,) ndarray optional
       Stationary vector on the full set of states. Assign zero
       stationary probabilities to states for which the
       stationary vector is unknown. Estimation will be made such
       that the resulting ensemble of transition matrices is
       defined on the intersection of the states with positive
       stationary vector and the largest connected set
       (undirected).

    count_mode : str, optional, default='effective'
       mode to obtain count matrices from discrete trajectories. Should be one of:

       * 'sliding' : A trajectory of length T will have :math:`T-\tau` counts
         at time indexes
         .. math:: (0 \rightarray \tau), (1 \rightarray \tau+1), ..., (T-\tau-1 \rightarray T-1)

       * 'effective' : Uses an estimate of the transition counts that are
         statistically uncorrelated. Recommended when used with a
         Bayesian MSM.

       * 'sample' : A trajectory of length T will have :math:`T / \tau` counts
         at time indexes
         .. math:: (0 \rightarray \tau), (\tau \rightarray 2 \tau), ..., (((T/tau)-1) \tau \rightarray T)

    sparse : bool, optional, default = False
       If true compute count matrix, transition matrix and all derived
       quantities using sparse matrix algebra. In this case python sparse
       matrices will be returned by the corresponding functions instead of
       numpy arrays. This behavior is suggested for very large numbers of
       states (e.g. > 4000) because it is likely to be much more efficient.

    dt_traj : str, optional, default='1 step'
       Description of the physical time corresponding to the trajectory time
       step. May be used by analysis algorithms such as plotting tools to
       pretty-print the axes. By default '1 step', i.e. there is no physical
       time unit. Specify by a number, whitespace and unit. Permitted units
       are (* is an arbitrary string):

       |  'fs',  'femtosecond*'
       |  'ps',  'picosecond*'
       |  'ns',  'nanosecond*'
       |  'us',  'microsecond*'
       |  'ms',  'millisecond*'
       |  's',   'second*'

    conf : float, optional, default=0.95
       Confidence interval. By default one-sigma (68.3%) is used. Use 95.4%
       for two sigma or 99.7% for three sigma.

    mincount_connectivity : float or '1/n'
       minimum number of counts to consider a connection between two states.
       Counts lower than that will count zero in the connectivity check and
       may thus separate the resulting transition matrix. The default
       evaluates to 1/nstates.

    References
    ----------
    .. [1] Trendelkamp-Schroer, B., H. Wu, F. Paul and F. Noe: Estimation and
       uncertainty of reversible Markov models. J. Chem. Phys.
       J. Chem. Phys. 143, 174101 (2015); https://doi.org/10.1063/1.4934536
    """

    def __init__(self, lagtime=1, nsamples=100, nsteps=None, reversible=True,
                 statdist_constraint=None, count_mode='effective', sparse=False,
                 dt_traj='1 step', conf=0.95,
                 maxiter=1000000,
                 maxerr=1e-8,
                 mincount_connectivity='1/n'):

        super(BayesianMSM, self).__init__(lagtime=lagtime, reversible=reversible,
                                          count_mode=count_mode, sparse=sparse,
                                          dt_traj=dt_traj,
                                          mincount_connectivity=mincount_connectivity)
        self.statdist_constraint = statdist_constraint
        self.maxiter = maxiter
        self.maxerr = maxerr
        self.nsamples = nsamples
        self.nsteps = nsteps
        self.conf = conf

    def fit(self, dtrajs, call_back: typing.Callable = None):
        """

        Parameters
        ----------
        dtrajs : list containing ndarrays(dtype=int) or ndarray(n, dtype=int)
            discrete trajectories, stored as integer ndarrays (arbitrary size)
            or a single ndarray for only one trajectory.

        call_back: callable or None (optional)
            function to be called to indicate progress of sampling.

        """
        # conduct MLE estimation (superclass) first
        super(BayesianMSM, self).fit(dtrajs)
        mle = MaximumLikelihoodMSM(lagtime=self.lagtime, reversible=self.reversible,
                                   statdist_constraint=self.statdist_constraint, count_mode=self.count_mode,
                                   sparse=self.sparse,
                                   dt_traj=self.dt_traj, mincount_connectivity=self.mincount_connectivity,
                                   maxiter=self.maxiter, maxerr=self.maxerr).fit(dtrajs).fetch_model()

        # transition matrix sampler
        from msmtools.estimation import tmatrix_sampler
        from math import sqrt
        if self.nsteps is None:
            self.nsteps = int(sqrt(mle.count_model.nstates))  # heuristic for number of steps to decorrelate
        # use the same count matrix as the MLE. This is why we have effective as a default
        if self.statdist_constraint is None:
            tsampler = tmatrix_sampler(mle.count_model.count_matrix_active, reversible=self.reversible,
                                       T0=mle.transition_matrix, nsteps=self.nsteps)
        else:
            # Use the stationary distribution on the active set of states
            statdist_active = mle.stationary_distribution
            # We can not use the MLE as T0. Use the initialization in the reversible pi sampler
            tsampler = tmatrix_sampler(mle.count_model.count_matrix_active, reversible=self.reversible,
                                       mu=statdist_active, nsteps=self.nsteps)

        sample_Ps, sample_mus = tsampler.sample(nsamples=self.nsamples, return_statdist=True, call_back=call_back)
        # construct sampled MSMs
        samples = [
            MarkovStateModel(P, pi=pi, reversible=self.reversible, dt_model=mle.dt_model, count_model=mle.count_model)
            for P, pi in zip(sample_Ps, sample_mus)
        ]

        self._model = BayesianPosterior(prior=mle, samples=samples)

        return self
