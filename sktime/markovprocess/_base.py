import numpy as np

from sktime.base import Estimator, Model
from sktime.markovprocess import Q_
from sktime.markovprocess.markov_state_model import MarkovStateModel


class EstimatedMSM(Model, MarkovStateModel):
    pass


# TODO: do not store dtrajs
# TODO: distinguish more between model and estimator attributes.

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

                (0 \rightarrow \tau), (\tau \rightarrow 2 \tau), ..., (((T/tau)-1) \tau \rightarrow T)

    sparse : bool, optional, default = False
        If true compute count matrix, transition matrix and all derived
        quantities using sparse matrix algebra. In this case python sparse
        matrices will be returned by the corresponding functions instead of
        numpy arrays. This behavior is suggested for very large numbers of
        states (e.g. > 4000) because it is likely to be much more efficient.
    connectivity : str, optional, default = 'largest'
        Connectivity mode. Three methods are intended (currently only 'largest'
        is implemented)

        * 'largest' : The active set is the largest reversibly connected set.
          All estimation will be done on this subset and all quantities
          (transition matrix, stationary distribution, etc) are only defined
          on this subset and are correspondingly smaller than the full set
          of states
        * 'all' : The active set is the full set of states. Estimation will be
          conducted on each reversibly connected set separately. That means
          the transition matrix will decompose into disconnected submatrices,
          the stationary vector is only defined within subsets, etc.
          Currently not implemented.
        * 'none' : The active set is the full set of states. Estimation will
          be conducted on the full set of
          states without ensuring connectivity. This only permits
          non-reversible estimation. Currently not implemented.

    dt_traj : str, optional, default='1 step'
        Description of the physical time of the input trajectories. May be used
        by analysis algorithms such as plotting tools to pretty-print the axes.
        By default '1 step', i.e. there is no physical time unit. Specify by a
        number, whitespace and unit. Permitted units are (* is an arbitrary
        string). E.g. 200 picoseconds or 200ps.

    score_method : str, optional, default='VAMP2'
        Score to be used with score function - see there for documentation.

        *  'VAMP1'  Sum of singular values of the symmetrized transition matrix.
        *  'VAMP2'  Sum of squared singular values of the symmetrized transition matrix.

    score_k : int or None
        The maximum number of eigenvalues or singular values used in the
        score. If set to None, all available eigenvalues will be used.

    mincount_connectivity : float or '1/n'
        minimum number of counts to consider a connection between two states.
        Counts lower than that will count zero in the connectivity check and
        may thus separate the resulting transition matrix. The default
        evaluates to 1/nstates.

    """

    def __init__(self, lagtime=1, reversible=True, count_mode='sliding', sparse=False,
                 connectivity='largest', dt_traj='1 step', score_method='VAMP2', score_k=10,
                 mincount_connectivity='1/n'):
        super(_MSMBaseEstimator, self).__init__()
        self._dtrajs_full = ()
        self._connected_sets = ()
        self.lagtime = lagtime

        # set basic parameters
        self.reversible = reversible

        # sparse matrix computation wanted?
        self.sparse = sparse

        # store counting mode (lowercase)
        self.count_mode = count_mode
        if self.count_mode not in ('sliding', 'effective', 'sample'):
            raise ValueError('count mode ' + count_mode + ' is unknown.')

        # store connectivity mode (lowercase)
        self.connectivity = connectivity

        # time step
        self.dt_traj = dt_traj

        # score
        self.score_method = score_method
        self.score_k = score_k

        # connectivity
        self.mincount_connectivity = mincount_connectivity

        self._C_full = None
        self._C_active = None
        self._nstates_full = None
        self._nstates_active = None
        self.active_set = None
        self._nstates_active = None
        self._nstates_full = None

    ################################################################################
    # Generic functions
    ################################################################################

    def _get_dtraj_stats(self, dtrajs):
        """ Compute raw trajectory counts

        Parameters
        ----------
        dtrajs : list containing ndarrays(dtype=int) or ndarray(n, dtype=int) or :class:`DiscreteTrajectoryStats <pyemma.msm.estimators._dtraj_stats.DiscreteTrajectoryStats>`
            discrete trajectories, stored as integer ndarrays (arbitrary size)
            or a single ndarray for only one trajectory.

        """
        # harvest discrete statistics
        from sktime.markovprocess._dtraj_stats import DiscreteTrajectoryStats
        if isinstance(dtrajs, DiscreteTrajectoryStats):
            dtrajstats = dtrajs
        else:
            # compute and store discrete trajectory statistics
            dtrajstats = DiscreteTrajectoryStats(dtrajs)
            # check if this MarkovStateModel seems too large to be dense
            if dtrajstats.nstates > 4000 and not self.sparse:
                import warnings
                warnings.warn('Building a dense MarkovStateModel with {nstates} states. This can be '
                              'inefficient or unfeasible in terms of both runtime and memory consumption. '
                              'Consider using sparse=True.'.format(nstates=dtrajstats.nstates))

        # count lagged
        dtrajstats.count_lagged(self.lagtime, count_mode=self.count_mode,
                                mincount_connectivity=self.mincount_connectivity,
                                n_jobs=getattr(self, 'n_jobs', None))

        # for other statistics
        return dtrajstats

    ################################################################################
    # Basic attributes
    ################################################################################

    @property
    def lagtime(self) -> Q_:
        """ The lag time at which the Markov model was estimated."""
        return self._lag

    @lagtime.setter
    def lagtime(self, value: [int, str]):
        self._lag = Q_(value)

    @property
    def nstates_full(self):
        r""" Number of states in discrete trajectories """
        return self._nstates_full

    @property
    def active_set(self):
        """
        The active set of states on which all computations and estimations will be done

        """
        return self._active_set

    @active_set.setter
    def active_set(self, value):
        self._active_set = value

    @property
    def connectivity(self):
        """Returns the connectivity mode of the MarkovStateModel """
        return self._connectivity

    @connectivity.setter
    def connectivity(self, value):
        if value == 'largest':
            pass  # this is the current default. no need to do anything
        elif value == 'all':
            raise NotImplementedError('MSM estimation with connectivity=\'all\' is currently not implemented.')
        elif value == 'none':
            raise NotImplementedError('MSM estimation with connectivity=\'none\' is currently not implemented.')
        else:
            raise ValueError('connectivity mode {} is unknown. Currently only "largest" is implemented'.format(value))
        self._connectivity = value

    @property
    def dt_traj(self) -> Q_:
        """Time interval between discrete steps of the time series."""
        return self.timestep_traj

    @dt_traj.setter
    def dt_traj(self, value: [int, str]):
        self.timestep_traj = Q_(value)

    @property
    def largest_connected_set(self):
        """The largest reversible connected set of states."""
        return self._connected_sets[0]

    @property
    def connected_sets(self):
        """The reversible connected sets of states, sorted by size (descending)."""
        return self._connected_sets

    @property
    def discrete_trajectories_full(self):
        """
        A list of integer arrays with the original (unmapped) discrete trajectories:

        """
        return self._dtrajs_full

    @property
    def discrete_trajectories_active(self):
        """
        A list of integer arrays with the discrete trajectories mapped to the connectivity mode used.
        For example, for connectivity='largest', the indexes will be given within the connected set.
        Frames that are not in the connected set will be -1.
        """
        # compute connected dtrajs
        self._dtrajs_active = []
        for dtraj in self._dtrajs_full:
            self._dtrajs_active.append(self._full2active[dtraj])

        return self._dtrajs_active

    @property
    def count_matrix_active(self):
        """The count matrix on the active set given the connectivity mode used.

        For example, for connectivity='largest', the count matrix is given only on the largest reversibly connected set.
        Attention: This count matrix has been obtained by sliding a window of length tau across the data. It contains
        a factor of tau more counts than are statistically uncorrelated. It's fine to use this matrix for maximum
        likelihood estimated, but it will give far too small errors if you use it for uncertainty calculations. In order
        to do uncertainty calculations, use the effective count matrix, see:
        :attr:`effective_count_matrix`

        See Also
        --------
        effective_count_matrix
            For a count matrix with effective (statistically uncorrelated) counts.

        """
        return self._C_active

    @property
    def count_matrix_full(self):
        """
        The count matrix on full set of discrete states, irrespective as to whether they are connected or not.
        Attention: This count matrix has been obtained by sliding a window of length tau across the data. It contains
        a factor of tau more counts than are statistically uncorrelated. It's fine to use this matrix for maximum
        likelihood estimated, but it will give far too small errors if you use it for uncertainty calculations. In order
        to do uncertainty calculations, use the effective count matrix, see: :attr:`effective_count_matrix`
        (only implemented on the active set), or divide this count matrix by tau.

        See Also
        --------
        effective_count_matrix
            For a active-set count matrix with effective (statistically uncorrelated) counts.

        """
        return self._C_full

    @property
    def active_state_fraction(self):
        """The fraction of states in the largest connected set.

        """
        return float(self._nstates) / float(self._nstates_full)

    @property
    def active_count_fraction(self):
        """The fraction of counts in the largest connected set.

        """
        from pyemma.util.discrete_trajectories import count_states

        hist = count_states(self._dtrajs_full)
        hist_active = hist[self.active_set]
        return float(np.sum(hist_active)) / float(np.sum(hist))

    ################################################################################
    # Generation of trajectories and samples
    ################################################################################

    @property
    def active_state_indexes(self):
        """
        Ensures that the connected states are indexed and returns the indices
        """
        if not hasattr(self, '_active_state_indexes'):
            from pyemma.util.discrete_trajectories import index_states
            self._active_state_indexes = index_states(self.discrete_trajectories_active)
        return self._active_state_indexes

    def generate_traj(self, N, start=None, stop=None, stride=1):
        """Generates a synthetic discrete trajectory of length N and simulation time stride * lag time * N

        This information can be used
        in order to generate a synthetic molecular dynamics trajectory - see
        :func:`pyemma.coordinates.save_traj`

        Note that the time different between two samples is the Markov model lag time tau. When comparing
        quantities computing from this synthetic trajectory and from the input trajectories, the time points of this
        trajectory must be scaled by the lag time in order to have them on the same time scale.

        Parameters
        ----------
        N : int
            Number of time steps in the output trajectory. The total simulation time is stride * lag time * N
        start : int, optional, default = None
            starting state. If not given, will sample from the stationary distribution of P
        stop : int or int-array-like, optional, default = None
            stopping set. If given, the trajectory will be stopped before N steps
            once a state of the stop set is reached
        stride : int, optional, default = 1
            Multiple of lag time used as a time step. By default, the time step is equal to the lag time

        Returns
        -------
        indexes : ndarray( (N, 2) )
            trajectory and time indexes of the simulated trajectory. Each row consist of a tuple (i, t), where i is
            the index of the trajectory and t is the time index within the trajectory.
            Note that the time different between two samples is the Markov model lag time tau

        See also
        --------
        pyemma.coordinates.save_traj
            in order to save this synthetic trajectory as a trajectory file with molecular structures

        """
        # TODO: this is the only function left which does something time-related in a multiple of tau rather than dt.
        # TODO: we could generate dt-strided trajectories by sampling tau times from the current state, but that would
        # TODO: probably lead to a weird-looking trajectory. Maybe we could use a HMM to generate intermediate 'hidden'
        # TODO: frames. Anyway, this is a nontrivial issue.
        # generate synthetic states
        from msmtools.generation import generate_traj as _generate_traj

        syntraj = _generate_traj(self.transition_matrix, N, start=start, stop=stop, dt=stride)
        # result
        return sample_indexes_by_sequence(self.active_state_indexes, syntraj)

    def sample_by_state(self, nsample, subset=None, replace=True):
        """Generates samples of the connected states.

        For each state in the active set of states, generates nsample samples with trajectory/time indexes.
        This information can be used in order to generate a trajectory of length nsample * nconnected using
        :func:`pyemma.coordinates.save_traj` or nconnected trajectories of length nsample each using
        :func:`pyemma.coordinates.save_traj`

        Parameters
        ----------
        nsample : int
            Number of samples per state. If replace = False, the number of returned samples per state could be smaller
            if less than nsample indexes are available for a state.
        subset : ndarray((n)), optional, default = None
            array of states to be indexed. By default all states in the connected set will be used
        replace : boolean, optional
            Whether the sample is with or without replacement

        Returns
        -------
        indexes : list of ndarray( (N, 2) )
            list of trajectory/time index arrays with an array for each state.
            Within each index array, each row consist of a tuple (i, t), where i is
            the index of the trajectory and t is the time index within the trajectory.

        See also
        --------
        pyemma.coordinates.save_traj
            in order to save the sampled frames sequentially in a trajectory file with molecular structures
        pyemma.coordinates.save_trajs
            in order to save the sampled frames in nconnected trajectory files with molecular structures

        """
        # generate connected state indexes
        return sample_indexes_by_state(self.active_state_indexes, nsample, subset=subset, replace=replace)

    # TODO: add sample_metastable() for sampling from metastable (pcca or hmm) states.
    def sample_by_distributions(self, distributions, nsample):
        """Generates samples according to given probability distributions

        Parameters
        ----------
        distributions : list or array of ndarray ( (n) )
            m distributions over states. Each distribution must be of length n and must sum up to 1.0
        nsample : int
            Number of samples per distribution. If replace = False, the number of returned samples per state could be
            smaller if less than nsample indexes are available for a state.

        Returns
        -------
        indexes : length m list of ndarray( (nsample, 2) )
            List of the sampled indices by distribution.
            Each element is an index array with a number of rows equal to nsample, with rows consisting of a
            tuple (i, t), where i is the index of the trajectory and t is the time index within the trajectory.

        """
        # generate connected state indexes
        return sample_indexes_by_distribution(self.active_state_indexes, distributions, nsample)

    ################################################################################
    # For general statistics
    ################################################################################
    def trajectory_weights(self):
        r"""Uses the MarkovStateModel to assign a probability weight to each trajectory frame.

        This is a powerful function for the calculation of arbitrary observables in the trajectories one has
        started the analysis with. The stationary probability of the MarkovStateModel will be used to reweigh all states.
        Returns a list of weight arrays, one for each trajectory, and with a number of elements equal to
        trajectory frames. Given :math:`N` trajectories of lengths :math:`T_1` to :math:`T_N`, this function
        returns corresponding weights:

        .. math::

            (w_{1,1}, ..., w_{1,T_1}), (w_{N,1}, ..., w_{N,T_N})

        that are normalized to one:

        .. math::

            \sum_{i=1}^N \sum_{t=1}^{T_i} w_{i,t} = 1

        Suppose you are interested in computing the expectation value of a function :math:`a(x)`, where :math:`x`
        are your input configurations. Use this function to compute the weights of all input configurations and
        obtain the estimated expectation by:

        .. math::

            \langle a \rangle = \sum_{i=1}^N \sum_{t=1}^{T_i} w_{i,t} a(x_{i,t})

        Or if you are interested in computing the time-lagged correlation between functions :math:`a(x)` and
        :math:`b(x)` you could do:

        .. math::

            \langle a(t) b(t+\tau) \rangle_t = \sum_{i=1}^N \sum_{t=1}^{T_i} w_{i,t} a(x_{i,t}) a(x_{i,t+\tau})


        Returns
        -------
        weights : list of ndarray
            The normalized trajectory weights. Given :math:`N` trajectories of lengths :math:`T_1` to :math:`T_N`,
            returns the corresponding weights:

            .. math::

                (w_{1,1}, ..., w_{1,T_1}), (w_{N,1}, ..., w_{N,T_N})

        """
        # compute stationary distribution, expanded to full set
        statdist_full = np.zeros([self._nstates_full])
        statdist_full[self.active_set] = self.stationary_distribution
        # histogram observed states
        import msmtools.dtraj as msmtraj
        hist = 1.0 * msmtraj.count_states(self.discrete_trajectories_full)
        # simply read off stationary distribution and accumulate total weight
        W = []
        wtot = 0.0
        for dtraj in self.discrete_trajectories_full:
            w = statdist_full[dtraj] / hist[dtraj]
            W.append(w)
            wtot += np.sum(w)
        # normalize
        for w in W:
            w /= wtot
        # done
        return W

    ################################################################################
    # HMM-based coarse graining
    ################################################################################

    def hmm(self, nhidden: int):
        """Estimates a hidden Markov state model as described in [1]_

        Parameters
        ----------
        nhidden : int
            number of hidden (metastable) states

        Returns
        -------
        hmsm : :class:`MaximumLikelihoodHMSM`

        References
        ----------
        .. [1] F. Noe, H. Wu, J.-H. Prinz and N. Plattner:
            Projected and hidden Markov models for calculating kinetics and metastable states of complex molecules
            J. Chem. Phys. 139, 184114 (2013)

        """
        # check if the time-scale separation is OK
        # if hmm.nstates = msm.nstates there is no problem. Otherwise, check spectral gap
        if self.nstates > nhidden:
            timescale_ratios = self.timescales()[:-1] / self.timescales()[1:]
            if timescale_ratios[nhidden - 2] < 1.5:
                self.logger.warning('Requested coarse-grained model with {nhidden} metastable states at lag={lag}.'
                                    ' The ratio of relaxation timescales between'
                                    ' {nhidden} and {nhidden_1} states is only {ratio}'
                                    ' while we recommend at least 1.5.'
                                    ' It is possible that the resulting HMM is inaccurate. Handle with caution.'.format(
                    lag=self.lagtime,
                    nhidden=nhidden,
                    nhidden_1=nhidden + 1,
                    ratio=timescale_ratios[nhidden - 2],
                ))
        # run HMM estimate
        from pyemma.msm.estimators.maximum_likelihood_hmsm import MaximumLikelihoodHMSM
        estimator = MaximumLikelihoodHMSM(lag=self.lagtime, nstates=nhidden, msm_init=self,
                                          reversible=self.reversible, dt_traj=self.dt_traj)
        estimator.estimate(self.discrete_trajectories_full)
        return estimator.model

    def coarse_grain(self, ncoarse: int, method='hmm') -> Model:
        r"""Returns a coarse-grained Markov model.

        Currently only the HMM method described in [1]_ is available for coarse-graining MSMs.

        Parameters
        ----------
        ncoarse : int
            number of coarse states

        Returns
        -------
        hmsm : :class:`MaximumLikelihoodHMSM`

        References
        ----------
        .. [1] F. Noe, H. Wu, J.-H. Prinz and N. Plattner:
            Projected and hidden Markov models for calculating kinetics and metastable states of complex molecules
            J. Chem. Phys. 139, 184114 (2013)

        """
        # check input
        assert _types.is_int(self.nstates) and 1 < ncoarse <= self.nstates, \
            'nstates must be an int in [2,msmobj.nstates]'

        return self.hmm(ncoarse)

    ################################################################################
    # MODEL VALIDATION
    ################################################################################

    def cktest(self, nsets, memberships=None, mlags=10, conf=0.95, err_est=False,
               n_jobs=None, show_progress=True):
        """ Conducts a Chapman-Kolmogorow test.

        Parameters
        ----------
        nsets : int
            number of sets to test on
        memberships : ndarray(nstates, nsets), optional
            optional state memberships. By default (None) will conduct a cktest
            on PCCA (metastable) sets.
        mlags : int or int-array, optional
            multiples of lag times for testing the Model, e.g. range(10).
            A single int will trigger a range, i.e. mlags=10 maps to
            mlags=range(10). The setting None will choose mlags automatically
            according to the longest available trajectory
        conf : float, optional
            confidence interval
        err_est : bool, optional
            compute errors also for all estimations (computationally expensive)
            If False, only the prediction will get error bars, which is often
            sufficient to validate a model.
        n_jobs : int, default=None
            how many jobs to use during calculation
        show_progress : bool, optional
            Show progress bars for calculation?

        Returns
        -------
        cktest : :class:`ChapmanKolmogorovValidator <pyemma.msm.ChapmanKolmogorovValidator>`


        References
        ----------
        This test was suggested in [1]_ and described in detail in [2]_.

        .. [1] F. Noe, Ch. Schuette, E. Vanden-Eijnden, L. Reich and
            T. Weikl: Constructing the Full Ensemble of Folding Pathways
            from Short Off-Equilibrium Simulations.
            Proc. Natl. Acad. Sci. USA, 106, 19011-19016 (2009)
        .. [2] Prinz, J H, H Wu, M Sarich, B Keller, M Senne, M Held, J D
            Chodera, C Schuette and F Noe. 2011. Markov models of
            molecular kinetics: Generation and validation. J Chem Phys
            134: 174105

        """
        if memberships is None:
            self.pcca(nsets)
            memberships = self.metastable_memberships
        ck = ChapmanKolmogorovValidator(self, self, memberships, mlags=mlags, conf=conf,
                                        n_jobs=n_jobs, err_est=err_est, show_progress=show_progress)
        ck.estimate(self._dtrajs_full)
        return ck
