import warnings
from typing import Optional

import numpy as np
from scipy.sparse import issparse
from ..tools.estimation import effective_count_matrix
from .._base import _MSMBaseEstimator
from . import MarkovStateModel
from .. import TransitionCountEstimator, TransitionCountModel, count_states

from . import _koopman_reweighted_msm_impl as _impl
from ...util.matrix import submatrix
from ...util.types import ensure_dtraj_list

__author__ = 'Feliks Nüske, Fabian Paul, marscher, clonker'


class KoopmanReweightedMSM(MarkovStateModel):
    r""" This class belongs to a markov state model which was estimated by Koopman reweighting.

    Parameters
    ----------
    transition_matrix : (n, n) ndarray
        The transition matrix, see :meth:`MarkovStateModel.__init__` .
    stationary_distribution : ndarray(n), optional, default=None
        Stationary distribution if already computed, see :meth:`MarkovStateModel.__init__` .
    reversible : bool, optional, default=None
        Whether MSM is reversible, see :meth:`MarkovStateModel.__init__` .
    n_eigenvalues : int, optional, default=None
        The number of eigenvalues / eigenvectors to be kept, see :meth:`MarkovStateModel.__init__` .
    ncv : int optional, default=None
        Performance parameter for reversible MSMs, see :meth:`MarkovStateModel.__init__` .
    count_model : TransitionCountModel, optional, default=None
        In case the model was estimated with a :class:`OOMReweightedMSM` estimator, this contains a count matrix
        based on lagged data, i.e., data with the last :code:`lag` frames removed and histogram information
        based on the full dtrajs.
    twostep_count_matrices : (n, n, n) ndarray, optional, default=None
        Two-step count matrices for all states, each :code:`twostep_count_matrices[:, i, :]` is a count matrix
        for each :math:`i=1,\ldots,n`.
    oom_components : (m, n, m) ndarray, optional, default=None
        Matrix of set-observable operators, where :code:`m` is the rank based on the rank decision made during
        estimation.
    oom_eigenvalues : (m,) ndarray, optional, default=None
        The eigenvalues from OOM.
    oom_evaluator : (m,) ndarray, optional, default=None
        Evaluator of OOM.
    oom_information_state_vector : (m,) ndarray, optional, default=None
        Information state vector of OOM.

    See Also
    --------
    MaximumLikelihoodMSM
    BayesianMSM
    """

    def __init__(self, transition_matrix: np.ndarray, stationary_distribution: Optional[np.ndarray] = None,
                 reversible: Optional[bool] = None, n_eigenvalues: Optional[int] = None, ncv: Optional[int] = None,
                 twostep_count_matrices: Optional[np.ndarray] = None, oom_components: Optional[np.ndarray] = None,
                 count_model: Optional[TransitionCountModel] = None, oom_eigenvalues: Optional[np.ndarray] = None,
                 oom_evaluator: Optional[np.ndarray] = None, oom_information_state_vector: Optional[np.ndarray] = None):
        super(KoopmanReweightedMSM, self).__init__(
            transition_matrix, stationary_distribution=stationary_distribution, reversible=reversible,
            n_eigenvalues=n_eigenvalues, ncv=ncv, count_model=count_model
        )
        self._oom_eigenvalues = oom_eigenvalues
        self._oom_components = oom_components
        self._oom_information_state_vector = oom_information_state_vector
        self._oom_evaluator = oom_evaluator
        self._twostep_count_matrices = twostep_count_matrices
        if oom_evaluator is not None:
            self._oom_rank = oom_evaluator.size

    @property
    def twostep_count_matrices(self):
        r""" Two-step count matrices for all states. C2t[:, n, :] is a count matrix for each n. """
        return self._twostep_count_matrices

    @property
    def oom_eigenvalues(self):
        """System eigenvalues estimated by OOM."""
        return self._oom_eigenvalues

    @property
    def oom_timescales(self):
        """System timescales estimated by OOM."""
        return -self.lagtime / np.log(np.abs(self._oom_eigenvalues[1:]))

    @property
    def oom_rank(self):
        """Return OOM model rank."""
        return self._oom_rank

    @property
    def oom_components(self):
        """Return OOM components."""
        return self._oom_components

    @property
    def oom_information_state_vector(self):
        """ Return OOM initial state vector."""
        return self._oom_information_state_vector

    @property
    def oom_evaluator(self):
        """Return OOM evaluator vector."""
        return self._oom_evaluator


class OOMReweightedMSM(_MSMBaseEstimator):
    r"""OOM (observable operator model) MSM estimator for MSMs given discrete trajectory statistics.
    Here, each transition is re-weighted using OOM theory. Details can be found in :footcite:`nuske2017markov`.

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
    .. footbibliography::
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
        r""" Yields the estimated Koopman reweighted MSM.

        Returns
        -------
        model : KoopmanReweightedMSM or None
            The estimated Koopman reweighted MSM or None if fit was not called yet.
        """
        return self._model

    def fit(self, dtrajs, **kw):
        r""" Fits an MSM using Koopman reweighting.

        Parameters
        ----------
        dtrajs : array_like or list of array_like
            Discrete trajectories.
        **kw
            API compatibility to sklearn, not actually used in algorithm.
        """
        dtrajs = ensure_dtraj_list(dtrajs)
        # remove last lag steps from dtrajs:
        dtrajs_lag = [traj[:-self.lagtime] for traj in dtrajs]

        # statistics are collected over full trajectories
        histogram = count_states(dtrajs, ignore_negative=True)
        # because of double counting, only count lagged trajs
        count_matrix = TransitionCountEstimator.count(count_mode=self.count_mode, dtrajs=dtrajs_lag,
                                                      lagtime=self.lagtime, sparse=self.sparse)
        count_model = TransitionCountModel(count_matrix, counting_mode=self.count_mode, lagtime=self.lagtime,
                                           state_histogram=histogram)
        count_model = count_model.submodel_largest(connectivity_threshold=self.connectivity_threshold, directed=True)

        # Estimate transition matrix using re-sampling:
        if self.rank_mode == 'bootstrap_counts':
            effective_count_mat = effective_count_matrix(dtrajs_lag, self.lagtime)
            Ceff = submatrix(effective_count_mat, count_model.state_symbols)
            smean, sdev = _impl.bootstrapping_count_matrix(Ceff, nbs=self.nbs)
        else:
            smean, sdev = _impl.bootstrapping_dtrajs(dtrajs_lag, self.lagtime, count_model.n_states_full, nbs=self.nbs,
                                                     active_set=count_model.state_symbols)
        # Estimate two step count matrices:
        twostep_count_matrices = _impl.twostep_count_matrix(dtrajs, self.lagtime, count_model.n_states_full)
        # Rank decision:
        rank_ind = _impl.rank_decision(smean, sdev, tol=self.tol_rank)
        # Estimate OOM components:
        if issparse(count_model.count_matrix_full):
            cmat = count_model.count_matrix_full.toarray()
        else:
            cmat = count_model.count_matrix_full
        oom_components, omega, sigma, eigenvalues = _impl.oom_components(
            cmat, twostep_count_matrices, rank_ind=rank_ind, lcc=count_model.state_symbols
        )
        # Compute transition matrix:
        P, lcc_new = _impl.equilibrium_transition_matrix(oom_components, omega, sigma, reversible=self.reversible)

        # Update active set and derived quantities:
        if lcc_new.size < count_model.n_states:
            assert isinstance(count_model, TransitionCountModel)
            count_model = count_model.submodel(count_model.symbols_to_states(lcc_new))
            warnings.warn("Caution: Re-estimation of count matrix resulted in reduction of the active set.")

        self._model = KoopmanReweightedMSM(
            transition_matrix=P,
            oom_eigenvalues=eigenvalues,
            oom_evaluator=sigma,
            oom_information_state_vector=omega,
            count_model=count_model,
            oom_components=oom_components,
            twostep_count_matrices=twostep_count_matrices
        )

        return self
