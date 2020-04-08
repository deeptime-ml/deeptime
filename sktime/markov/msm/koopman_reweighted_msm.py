import warnings

import numpy as _np
from msmtools.estimation import effective_count_matrix
from scipy.sparse import issparse
from sktime.markov._base import _MSMBaseEstimator
from sktime.markov.msm import MarkovStateModel
from sktime.markov.transition_counting import TransitionCountEstimator, TransitionCountModel
from sktime.util import submatrix

from . import _koopman_reweighted_msm_impl as _impl

__author__ = 'Feliks Nüske, Fabian Paul, marscher, clonker'


class KoopmanReweightedMSM(MarkovStateModel):
    def __init__(self, transition_matrix, c2t=None, oom_components=None,
                 stationary_distribution=None, reversible=None, n_eigenvalues=None, ncv=None,
                 count_model=None, eigenvalues_oom=None, sigma=None, omega=None):
        super(KoopmanReweightedMSM, self).__init__(
            transition_matrix, stationary_distribution=stationary_distribution, reversible=reversible,
            n_eigenvalues=n_eigenvalues, ncv=ncv, count_model=count_model
        )
        self._eigenvalues_oom = eigenvalues_oom
        self._oom_components = oom_components
        self._omega = omega
        self._sigma = sigma
        self._c2t = c2t
        if sigma is not None:
            self._oom_rank = sigma.size

    @property
    def twostep_count_matrix(self):
        r""" Two-step count matrices for all states. C2t[:, n, :] is a count matrix for each n. """
        return self._c2t

    @property
    def eigenvalues_oom(self):
        """System eigenvalues estimated by OOM."""
        return self._eigenvalues_oom

    @property
    def timescales_oom(self):
        """System timescales estimated by OOM."""
        return -self.lagtime / _np.log(_np.abs(self._eigenvalues_oom[1:]))

    @property
    def oom_rank(self):
        """Return OOM model rank."""
        return self._oom_rank

    @property
    def oom_components(self):
        """Return OOM components."""
        return self._oom_components

    @property
    def oom_omega(self):
        """ Return OOM initial state vector."""
        return self._omega

    @property
    def oom_sigma(self):
        """Return OOM evaluator vector."""
        return self._sigma


class OOMReweightedMSM(_MSMBaseEstimator):
    r"""Maximum likelihood estimator for MSMs given discrete trajectory statistics.

    Parameters
    ----------
    lagtime : int
        lag time at which transitions are counted and the transition matrix is
        estimated.

    reversible : bool, optional, default = True
        If true compute reversible MSM, else non-reversible MSM

    count_mode : str, optional, default='sliding'
        mode to obtain count matrices from discrete trajectories. Should be
        one of:

        * 'sliding' : A trajectory of length T will have :math:`T-\tau` counts
          at time indexes

          .. math::

             (0 \rightarrow \tau), (1 \rightarrow \tau+1), ..., (T-\tau-1 \rightarrow T-1)
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

    time_unit : str, optional, default='1 step'
        Description of the physical time of the input trajectories. May be used
        by analysis algorithms such as plotting tools to pretty-print the axes.
        By default '1 step', i.e. there is no physical time unit. Specify by a
        number, whitespace and unit. Permitted units are (* is an arbitrary
        string):

        |  'fs',  'femtosecond*'
        |  'ps',  'picosecond*'
        |  'ns',  'nanosecond*'
        |  'us',  'microsecond*'
        |  'ms',  'millisecond*'
        |  's',   'second*'

    nbs : int, optional, default=10000
        number of re-samplings for rank decision in OOM estimation.

    rank_mode : str, optional
        Re-sampling method for model rank selection. Can be
        * 'bootstrap_counts': Directly re-sample transitions based on effective count matrix.

        * 'bootstrap_trajs': Re-draw complete trajectories with replacement.

    tol_rank: float, optional, default = 10.0
        signal-to-noise threshold for rank decision.

    connectivity_threshold : float or '1/n'
        minimum number of counts to consider a connection between two states.
        Counts lower than that will count zero in the connectivity check and
        may thus separate the resulting transition matrix. The default
        evaluates to 1/n_states.

    References
    ----------
    .. [1] H. Wu and F. Noe: Variational approach for learning Markov processes from time series data
        (in preparation)

    """

    def __init__(self, lagtime, reversible=True, count_mode='sliding', sparse=False,
                 time_unit='1 step', nbs=10000, rank_mode='bootstrap_counts', tol_rank=10.0,
                 connectivity_threshold='1/n'):
        super(OOMReweightedMSM, self).__init__(reversible=reversible, sparse=sparse)
        # Check count mode:
        self.count_mode = str(count_mode).lower()
        if self.count_mode not in ('sliding', 'sample'):
            raise ValueError(f'count mode {count_mode} is unknown. Only \'sliding\' and \'sample\' are allowed.')
        if rank_mode not in ('bootstrap_counts', 'bootstrap_trajs'):
            raise ValueError('rank_mode must be either \'bootstrap_counts\' or \'bootstrap_trajs\'')
        self.nbs = nbs
        self.tol_rank = tol_rank
        self.rank_mode = rank_mode
        self.lagtime = lagtime
        self.time_unit = time_unit
        self.connectivity_threshold = connectivity_threshold

    def fetch_model(self) -> KoopmanReweightedMSM:
        return self._model

    def fit(self, dtrajs, **kw):
        # remove last lag steps from dtrajs:
        dtrajs_lag = [traj[:-self.lagtime] for traj in dtrajs]
        count_model = TransitionCountEstimator(lagtime=self.lagtime, count_mode=self.count_mode, sparse=self.sparse)\
            .fit(dtrajs_lag).fetch_model()
        count_model = count_model.submodel_largest(connectivity_threshold=self.connectivity_threshold)

        # Estimate transition matrix using re-sampling:
        if self.rank_mode == 'bootstrap_counts':
            effective_count_mat = effective_count_matrix(dtrajs_lag, self.lagtime)
            Ceff = submatrix(effective_count_mat, count_model.state_symbols)
            smean, sdev = _impl.bootstrapping_count_matrix(Ceff, nbs=self.nbs)
        else:
            smean, sdev = _impl.bootstrapping_dtrajs(dtrajs_lag, self.lagtime, count_model.n_states, nbs=self.nbs,
                                                     active_set=count_model.state_symbols)
        # Estimate two step count matrices:
        c2t = _impl.twostep_count_matrix(dtrajs, self.lagtime, count_model.n_states)
        # Rank decision:
        rank_ind = _impl.rank_decision(smean, sdev, tol=self.tol_rank)
        # Estimate OOM components:
        if issparse(count_model.count_matrix):
            cmat = count_model.count_matrix.toarray()
        else:
            cmat = count_model.count_matrix
        Xi, omega, sigma, eigenvalues = _impl.oom_components(cmat, c2t, rank_ind=rank_ind,
                                                             lcc=count_model.state_symbols)
        # Compute transition matrix:
        P, lcc_new = _impl.equilibrium_transition_matrix(Xi, omega, sigma, reversible=self.reversible)

        # Update active set and derived quantities:
        if lcc_new.size < count_model.n_states:
            assert isinstance(count_model, TransitionCountModel)
            count_model = count_model.submodel(count_model.symbols_to_states(lcc_new))
            warnings.warn("Caution: Re-estimation of count matrix resulted in reduction of the active set.")

        self._model = KoopmanReweightedMSM(
            transition_matrix=P,
            eigenvalues_oom=eigenvalues,
            sigma=sigma,
            omega=omega,
            count_model=count_model,
            oom_components=Xi,
            c2t=c2t
        )

        return self
