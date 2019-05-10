

import warnings

import numpy as _np
from msmtools import estimation as msmest

from pyemma.msm.estimators._OOM_MSM import bootstrapping_count_matrix, bootstrapping_dtrajs, twostep_count_matrix, \
    rank_decision, oom_components, equilibrium_transition_matrix
from pyemma.util.annotators import fix_docs, aliased

from sktime.markovprocess._base import _MSMBaseEstimator

__author__ = 'Feliks Nueske, Fabian Paul, marscher'


@fix_docs
@aliased
class OOMReweightedMSM(_MSMBaseEstimator):
    r"""Maximum likelihood estimator for MSMs given discrete trajectory statistics

    Parameters
    ----------
    lag : int
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

    dt_traj : str, optional, default='1 step'
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

    rank_Ct : str, optional
        Re-sampling method for model rank selection. Can be
        * 'bootstrap_counts': Directly re-sample transitions based on effective count matrix.

        * 'bootstrap_trajs': Re-draw complete trajectories with replacement.

    tol_rank: float, optional, default = 10.0
        signal-to-noise threshold for rank decision.

    mincount_connectivity : float or '1/n'
        minimum number of counts to consider a connection between two states.
        Counts lower than that will count zero in the connectivity check and
        may thus separate the resulting transition matrix. The default
        evaluates to 1/nstates.

    References
    ----------
    .. [1] H. Wu and F. Noe: Variational approach for learning Markov processes from time series data
        (in preparation)

    """
    def __init__(self, lagtime, reversible=True, count_mode='sliding', sparse=False,
                 dt_traj='1 step', nbs=10000, rank_Ct='bootstrap_counts', tol_rank=10.0,
                 mincount_connectivity='1/n'):

        # Check count mode:
        self.count_mode = str(count_mode).lower()
        if self.count_mode not in ('sliding', 'sample'):
            raise ValueError('count mode {} is unknown. Only \'sliding\' and \'sample\' are allowed.'.format(count_mode))
        if rank_Ct not in ('bootstrap_counts', 'bootstrap_trajs'):
            raise ValueError('rank_Ct must be either \'bootstrap_counts\' or \'bootstrap_trajs\'')

        super(OOMReweightedMSM, self).__init__(lagtime=lagtime, reversible=reversible, count_mode=count_mode, sparse=sparse,
                                               dt_traj=dt_traj, mincount_connectivity=mincount_connectivity)
        self.nbs = nbs
        self.tol_rank = tol_rank
        self.rank_Ct = rank_Ct

    def fit(self, dtrajs):
        # remove last lag steps from dtrajs:
        dtrajs_lag = [traj[:-self.lag] for traj in dtrajs]
        self._compute_count_matrix(dtrajs, mincount_connectivity=self.mincount_connectivity, count_mode=self.count_mode)

        # Estimate transition matrix using re-sampling:
        if self.rank_Ct == 'bootstrap_counts':
            Ceff_full = msmest.effective_count_matrix(dtrajs_lag, self.lag)
            from pyemma.util.linalg import submatrix
            Ceff = submatrix(Ceff_full, self.active_set)
            smean, sdev = bootstrapping_count_matrix(Ceff, nbs=self.nbs)
        else:
            smean, sdev = bootstrapping_dtrajs(dtrajs_lag, self.lag, self.nstates_full, nbs=self.nbs,
                                               active_set=self.active_set)
        # Estimate two step count matrices:
        C2t = twostep_count_matrix(dtrajs, self.lag, self.nstates_full)
        # Rank decision:
        rank_ind = rank_decision(smean, sdev, tol=self.tol_rank)
        # Estimate OOM components:
        Xi, omega, sigma, l = oom_components(self._C_full.toarray(), C2t, rank_ind=rank_ind,
                                             lcc=self.active_set)
        # Compute transition matrix:
        P, lcc_new = equilibrium_transition_matrix(Xi, omega, sigma, reversible=self.reversible)

        # Update active set and derived quantities:
        if lcc_new.size < self.nstates:
            self.active_set = self.active_set[lcc_new]
            self._C_active = self.count_matrix(subset=self.active_set)
            self._nstates = self._C_active.shape[0]
            self._full2lcs = -1 * _np.ones(len(self.C_active), dtype=int)
            self._full2lcs[self.active_set] = _np.arange(len(self.active_set))
            warnings.warn("Caution: Re-estimation of count matrix resulted in reduction of the active set.")

        # continue sparse or dense?
        if not self.sparse:
            # converting count matrices to arrays. As a result the
            # transition matrix and all subsequent properties will be
            # computed using dense arrays and dense matrix algebra.
            self._C_active = self._C_active.toarray()

        # Done. We set our own model parameters, so this estimator is
        # equal to the estimated model.

        # update model
        m = self._model
        m.transition_matrix = P
        m.reversible = self.reversible
        # TODO: should the model know about the connected sets?
        m._connected_sets = msmest.connected_sets(m._C_full)

        m._Xi = Xi
        m._omega = omega
        m._sigma = sigma
        m._eigenvalues_OOM = l
        m._rank_ind = rank_ind
        m._oom_rank = m._sigma.size
        m._C2t = C2t

        return self

    def _blocksplit_dtrajs(self, dtrajs, sliding):
        """ Override splitting method of base class.

        For OOM estimators we currently need a clean trajectory splitting, i.e. we don't do block splitting at all.

        """
        if len(dtrajs) < 2:
            raise NotImplementedError('Current cross-validation implementation for OOMReweightedMSM requires' +
                                      'multiple trajectories. You can split the trajectory yourself into training' +
                                      'and test set and use the score method after fitting the training set.')
        return dtrajs

    @property
    def eigenvalues_OOM(self):
        """
            System eigenvalues estimated by OOM.

        """
        self._check_is_estimated()
        return self._eigenvalues_OOM

    @property
    def timescales_OOM(self):
        """
            System timescales estimated by OOM.

        """
        self._check_is_estimated()
        return -self.lag / _np.log(_np.abs(self._eigenvalues_OOM[1:]))

    @property
    def OOM_rank(self):
        """
            Return OOM model rank.

        """
        self._check_is_estimated()
        return self._oom_rank

    @property
    def OOM_components(self):
        """
            Return OOM components.

        """
        self._check_is_estimated()
        return self._Xi

    @property
    def OOM_omega(self):
        """
            Return OOM initial state vector.

        """
        self._check_is_estimated()
        return self._omega

    @property
    def OOM_sigma(self):
        """
            Return OOM evaluator vector.

        """
        self._check_is_estimated()
        return self._sigma
