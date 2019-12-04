
# This file is part of PyEMMA.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# PyEMMA is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import warnings

import numpy as np
from msmtools.dtraj import number_of_states

from sktime.base import Estimator
from sktime.markovprocess import MarkovStateModel, transition_counting
from sktime.markovprocess.bhmm import discrete_hmm, lag_observations, init_discrete_hmm
from sktime.markovprocess.bhmm.init.discrete import init_discrete_hmm_spectral
from sktime.markovprocess.hidden_markov_model import HMSM
from sktime.util import ensure_dtraj_list


class MaximumLikelihoodHMSM(Estimator):
    r"""Maximum likelihood estimator for a Hidden MSM given a MSM

    Parameters
    ----------
    nstates : int, optional, default=2
        number of hidden states
    lag : int, optional, default=1
        lagtime to fit the HMSM at
    stride : str or int, default=1
        stride between two lagged trajectories extracted from the input
        trajectories. Given trajectory s[t], stride and lag will result
        in trajectories
            s[0], s[lag], s[2 lag], ...
            s[stride], s[stride + lag], s[stride + 2 lag], ...
        Setting stride = 1 will result in using all data (useful for maximum
        likelihood estimator), while a Bayesian estimator requires a longer
        stride in order to have statistically uncorrelated trajectories.
        Setting stride = 'effective' uses the largest neglected timescale as
        an fit for the correlation time and sets the stride accordingly
    msm_init : str or :class:`MSM <sktime.markovprocess.MarkovStateModel>`
        MSM object to initialize the estimation, or one of following keywords:

        * 'largest-strong' or None (default) : Estimate MSM on the largest
            strongly connected set and use spectral clustering to generate an
            initial HMM
        * 'all' : Estimate MSM(s) on the full state space to initialize the
            HMM. This fit maybe weakly connected or disconnected.
    reversible : bool, optional, default = True
        If true compute reversible MSM, else non-reversible MSM
    stationary : bool, optional, default=False
        If True, the initial distribution of hidden states is self-consistently computed as the stationary
        distribution of the transition matrix. If False, it will be estimated from the starting states.
        Only set this to true if you're sure that the observation trajectories are initiated from a global
        equilibrium distribution.
    connectivity : str, optional, default = None
        Defines if the resulting HMM will be defined on all hidden states or on
        a connected subset. Connectivity is defined by counting only
        transitions with at least mincount_connectivity counts.
        If a subset of states is used, all estimated quantities (transition
        matrix, stationary distribution, etc) are only defined on this subset
        and are correspondingly smaller than nstates.
        Following modes are available:

        * None or 'all' : The active set is the full set of states.
          Estimation is done on all weakly connected subsets separately. The
          resulting transition matrix may be disconnected.
        * 'largest' : The active set is the largest reversibly connected set.
        * 'populous' : The active set is the reversibly connected set with most counts.
    mincount_connectivity : float or '1/n'
        minimum number of counts to consider a connection between two states.
        Counts lower than that will count zero in the connectivity check and
        may thus separate the resulting transition matrix. The default
        evaluates to 1/nstates.
    separate : None or iterable of int
        Force the given set of observed states to stay in a separate hidden state.
        The remaining nstates-1 states will be assigned by a metastable decomposition.
    observe_nonempty : bool
        If True, will restricted the observed states to the states that have
        at least one observation in the lagged input trajectories.
        If an initial MSM is given, this option is ignored and the observed
        subset is always identical to the active set of that MSM.
    dt_traj : str, optional, default='1 step'
        Description of the physical time corresponding to the trajectory time
        step.  May be used by analysis algorithms such as plotting tools to
        pretty-print the axes. By default '1 step', i.e. there is no physical
        time unit. Specify by a number, whitespace and unit. Permitted units
        are (* is an arbitrary string):

        |  'fs',  'femtosecond*'
        |  'ps',  'picosecond*'
        |  'ns',  'nanosecond*'
        |  'us',  'microsecond*'
        |  'ms',  'millisecond*'
        |  's',   'second*'

    accuracy : float, optional, default = 1e-3
        convergence threshold for EM iteration. When two the likelihood does
        not increase by more than accuracy, the iteration is stopped
        successfully.
    maxit : int, optional, default = 1000
        stopping criterion for EM iteration. When so many iterations are
        performed without reaching the requested accuracy, the iteration is
        stopped without convergence (a warning is given)

    """
    def __init__(self, nstates=2, lagtime=1, stride=1, msm_init='largest-strong', reversible=True, stationary=False,
                 connectivity=None, mincount_connectivity='1/n', observe_nonempty=True, separate=None,
                 dt_traj='1 step', accuracy=1e-3, maxit=1000):
        super(MaximumLikelihoodHMSM, self).__init__()
        self.nstates = nstates
        self.lagtime = lagtime
        self.stride = stride
        self.msm_init = msm_init
        self.reversible = reversible
        self.stationary = stationary
        self.connectivity = connectivity
        if mincount_connectivity == '1/n':
            mincount_connectivity = 1.0 / float(nstates)
        self.mincount_connectivity = mincount_connectivity
        self.separate = separate
        self.observe_nonempty = observe_nonempty
        self.dt_traj = dt_traj
        self.accuracy = accuracy
        self.maxit = maxit

    def _create_model(self) -> HMSM:
        return HMSM()

    def fit(self, dtrajs, **kwargs):
        from .bhmm.estimators.maximum_likelihood import MaximumLikelihoodHMM

        # OBSERVATION SET
        observe_subset = 'nonempty' if self.observe_nonempty else None

        hmm_count_model = _HMMCountEstimator(nstates=self.nstates, lagtime=self.lagtime, stride=self.stride) \
            .fit(dtrajs).fetch_model()
        assert isinstance(hmm_count_model, _HMMTransitionCounts)
        dtrajs_lagged_strided = hmm_count_model.dtrajs_lagged_strided

        # INIT HMM
        if isinstance(self.msm_init, str):
            args = dict(observations=dtrajs_lagged_strided, nstates=self.nstates, lag=1,
                        reversible=self.reversible, stationary=True, regularize=True,
                        separate=self.separate)
            if self.msm_init == 'largest-strong':
                args['method'] = 'lcs-spectral'
            elif self.msm_init == 'all':
                args['method'] = 'spectral'

            hmm_init = init_discrete_hmm(**args)
        else:
            assert isinstance(self.msm_init, MarkovStateModel)
            msm_count_model = self.msm_init.count_model
            p0, P0, pobs0 = init_discrete_hmm_spectral(msm_count_model.count_matrix.toarray(), self.nstates,
                                                       reversible=self.reversible, stationary=True,
                                                       active_set=msm_count_model.active_set,
                                                       P=self.msm_init.transition_matrix, separate=self.separate)
            hmm_init = discrete_hmm(p0, P0, pobs0)
            observe_subset = msm_count_model.active_set  # override observe_subset.

        # ---------------------------------------------------------------------------------------
        # Estimate discrete HMM
        # ---------------------------------------------------------------------------------------
        hmm_est = MaximumLikelihoodHMM(self.nstates, initial_model=hmm_init,
                                       output='discrete', reversible=self.reversible, stationary=self.stationary,
                                       accuracy=self.accuracy, maxit=self.maxit)
        hmm = hmm_est.fit(dtrajs_lagged_strided).fetch_model()
        # update the count matrix from the counts obtained via the Viterbi paths.
        # TODO: this seems like a hack, but is due to the duality of HMM and HMSM models...
        # TODO: it is easy to introduce bugs due to invoking the ctor again (forgot a param), update_model_params of pyemma seems more reasonable for this case.
        hmm_count_model.__init__(stride=self.stride,
                                 count_matrix=hmm.transition_counts,
                                 initial_count=hmm.initial_count, lagtime=self.lagtime, dt_traj=self.dt_traj,
                                 nstates=self.nstates,
                                 nstates_obs=number_of_states(dtrajs_lagged_strided),
                                 active_set=None)
        # udpate model params would only set these:
        # hmm_count_model._update_params(count_matrix=hmm.transition_counts, initial_counts=hmm.initial_counts)
        # get estimation parameters
        self._model.__init__(transition_matrix=hmm.transition_matrix,
                             pobs=hmm.output_model.output_probabilities,
                             pi=hmm.stationary_distribution,
                             dt_model=hmm_count_model.dt_traj*self.lagtime,
                             reversible=self.reversible,
                             initial_distribution=hmm.initial_distribution, count_model=hmm_count_model)
        self._model.hmm = hmm  # TODO: duplicity...

        # TODO: perhaps remove connectivity and just rely on .submodel()?
        # deal with connectivity
        states_subset = None
        if self.connectivity == 'largest':
            states_subset = 'largest-strong'
        elif self.connectivity == 'populous':
            states_subset = 'populous-strong'

        # restrict model to observed and desired state space with given minimum count connectivity
        sub_model = self._model.submodel(states=states_subset, obs=observe_subset,
                                         mincount_connectivity=self.mincount_connectivity,
                                         inplace=True)
        # TODO:evil
        self._model = sub_model

        return self

    @property
    def msm_init(self):
        """ MSM initialization method, should be one of:
        * instance of :class:`MSM <sktime.markovprocess.MarkovStateModel>`

        or a string:

        * 'largest-strong' or None (default) : Estimate MSM on the largest
            strongly connected set and use spectral clustering to generate an
            initial HMM
        * 'all' : Estimate MSM(s) on the full state space to initialize the
            HMM. This fit maybe weakly connected or disconnected.
        """
        return self._msm_init

    @msm_init.setter
    def msm_init(self, value: [str, MarkovStateModel]):
        if isinstance(value, MarkovStateModel) and value.count_model is None:
            raise NotImplementedError('currently we obtain the active set and the count matrix from '
                                      'the provided count_model of the MSM.')
        elif isinstance(value, str):
            supported = ('largest-strong', 'all')
            if not value in supported:
                raise NotImplementedError(f'unknown msm_init value, was "{value}",'
                                          f'but valid options are {supported}.')
        self._msm_init = value

    @property
    def connectivity(self):
        return self._connectivity

    @connectivity.setter
    def connectivity(self, value):
        allowed = (None, 'largest', 'populus')
        if value not in allowed:
            raise ValueError(f'Illegal value for connectivity: {value}. Allowed values are one of: {allowed}.')
        self._connectivity = value

    # TODO: model attribute
    def compute_trajectory_weights(self, dtrajs_observed):
        r"""Uses the HMSM to assign a probability weight to each trajectory frame.

        This is a powerful function for the calculation of arbitrary observables in the trajectories one has
        started the analysis with. The stationary probability of the MSM will be used to reweigh all states.
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
        The normalized trajectory weights. Given :math:`N` trajectories of lengths :math:`T_1` to :math:`T_N`,
        returns the corresponding weights:

        .. math::

            (w_{1,1}, ..., w_{1,T_1}), (w_{N,1}, ..., w_{N,T_N})

        """
        # compute stationary distribution, expanded to full set
        statdist = self.stationary_distribution_obs
        statdist = np.append(statdist, [-1])  # add a zero weight at index -1, to deal with unobserved states
        # histogram observed states
        import msmtools.dtraj as msmtraj
        hist = 1.0 * msmtraj.count_states(dtrajs_observed, ignore_negative=True)
        # simply read off stationary distribution and accumulate total weight
        W = []
        wtot = 0.0
        for dtraj in self.discrete_trajectories_obs:
            w = statdist[dtraj] / hist[dtraj]
            W.append(w)
            wtot += np.sum(w)
        # normalize
        for w in W:
            w /= wtot
        # done
        return W

    ################################################################################
    # Generation of trajectories and samples
    ################################################################################

    # TODO: generate_traj. How should that be defined? Probably indexes of observable states, but should we specify
    #                      hidden or observable states as start and stop states?
    # TODO: sample_by_state. How should that be defined?

    def sample_by_observation_probabilities(self, nsample):
        r"""Generates samples according to the current observation probability distribution

        Parameters
        ----------
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
        from msmtools.dtraj import sample_indexes_by_distribution
        return sample_indexes_by_distribution(self.observable_state_indexes, self.observation_probabilities, nsample)

    ################################################################################
    # Model Validation
    ################################################################################

    def cktest(self, dtrajs, mlags=10, conf=0.95, err_est=False):
        """ Conducts a Chapman-Kolmogorow test.

        Parameters
        ----------
        dtrajs:
        mlags : int or int-array, default=10
            multiples of lag times for testing the Model, e.g. range(10).
            A single int will trigger a range, i.e. mlags=10 maps to
            mlags=range(10). The setting None will choose mlags automatically
            according to the longest available trajectory
        conf : float, optional, default = 0.95
            confidence interval
        err_est : bool, default=False
            compute errors also for all estimations (computationally expensive)
            If False, only the prediction will get error bars, which is often
            sufficient to validate a model.
        n_jobs : int, default=None
            how many jobs to use during calculation
        show_progress : bool, default=True
            Show progressbars for calculation?

        Returns
        -------
        cktest : :class:`ChapmanKolmogorovValidator <pyemma.msm.ChapmanKolmogorovValidator>`

        References
        ----------
        This is an adaption of the Chapman-Kolmogorov Test described in detail
        in [1]_ to Hidden MSMs as described in [2]_.

        .. [1] Prinz, J H, H Wu, M Sarich, B Keller, M Senne, M Held, J D
            Chodera, C Schuette and F Noe. 2011. Markov models of
            molecular kinetics: Generation and validation. J Chem Phys
            134: 174105

        .. [2] F. Noe, H. Wu, J.-H. Prinz and N. Plattner: Projected and hidden
            Markov models for calculating kinetics and metastable states of complex
            molecules. J. Chem. Phys. 139, 184114 (2013)

        """
        from sktime.markovprocess.chapman_kolmogorov_validator import ChapmanKolmogorovValidator
        try:
            model = self.fetch_model()
            if hasattr(model, 'prior'):
                model = model.prior
        except AttributeError:
            raise RuntimeError('call fit() first!')
        ck = ChapmanKolmogorovValidator(model, self, np.eye(self.nstates),
                                        mlags=mlags, conf=conf, err_est=err_est)
        ck.fit(dtrajs)
        return ck.fetch_model()


class _HMMTransitionCounts(transition_counting.TransitionCountModel):
    def __init__(self, nstates=None, nstates_obs=None, stride=1, initial_count=None, **kwargs):
        super(_HMMTransitionCounts, self).__init__(**kwargs)
        self._count_matrix_EM = None
        self.initial_count = initial_count
        self._nstates_full = nstates
        self._nstates_obs = nstates_obs
        if nstates_obs is not None:
            self._observable_set = np.arange(nstates_obs)
        self.stride = stride

    @property
    def initial_count(self):
        """ hidden init count """
        return self._initial_counts

    @initial_count.setter
    def initial_count(self, value):
        self._initial_counts = value

    @property
    def count_matrix(self):
        """ Hidden count matrix consistent with transition matrix """
        return super(_HMMTransitionCounts, self).count_matrix

    @property
    def count_matrix_EM(self):
        """ """
        return self._count_matrix_EM

    @property
    def dtrajs_lagged_strided(self):
        # TODO: we do not want this.
        return self._dtrajs

    @property
    def dtrajs_full(self):
        return self._dtrajs_full

    @property
    def nstates_obs(self):
        return self._nstates_obs

    @property
    def observable_set(self):
        return self._observable_set


class _HMMCountEstimator(transition_counting.TransitionCountEstimator):
    def __init__(self, lagtime, nstates, stride='effective'):
        super(_HMMCountEstimator, self).__init__()
        self.lagtime = lagtime
        self.stride = stride
        self.nstates = nstates

    def _create_model(self) -> _HMMTransitionCounts:
        return _HMMTransitionCounts()

    def fit(self, data):
        dtrajs = ensure_dtraj_list(data)
        # CHECK LAG
        trajlengths = [len(dtraj) for dtraj in dtrajs]
        if self.lagtime >= np.max(trajlengths):
            raise ValueError(f'Illegal lag time {self.lagtime} exceeds longest trajectory length')
        if self.lagtime > np.mean(trajlengths):
            warnings.warn(f'Lag time {self.lagtime} is on the order of mean trajectory length '
                          f'{np.mean(trajlengths)}. It is recommended to fit four lag times in each '
                          'trajectory. HMM might be inaccurate.')

        # EVALUATE STRIDE
        if self.stride == 'effective':
            stride = _HMMCountEstimator._compute_effective_stride(dtrajs, self.stride, self.lagtime, self.nstates)
        else:
            stride = self.stride

        # LAG AND STRIDE DATA
        dtrajs_lagged_strided = lag_observations(dtrajs, int(self.lagtime), stride=stride)
        nstates_obs = number_of_states(dtrajs_lagged_strided)
        # placeholder for counts computed from transition paths during HMM estimation.
        self._model.__init__(lagtime=self.lagtime, stride=stride, nstates_obs=nstates_obs)
        self._model._dtrajs = dtrajs_lagged_strided
        self._model._dtrajs_full = dtrajs

        return self

    @staticmethod
    def _compute_effective_stride(dtrajs, stride, lagtime, nstates):
        if stride != 'effective':
            raise RuntimeError('call this only if self.stride=="effective"!')
        # by default use lag as stride (=lag sampling), because we currently have no better theory for deciding
        # how many uncorrelated counts we can make
        stride = lagtime
        # get a quick fit from the spectral radius of the non-reversible
        from sktime.markovprocess import MaximumLikelihoodMSM
        msm_non_rev = MaximumLikelihoodMSM(lagtime=lagtime, reversible=False, sparse=False).fit(dtrajs).fetch_model()
        # if we have more than nstates timescales in our MSM, we use the next (neglected) timescale as an
        # fit of the de-correlation time
        if msm_non_rev.nstates > nstates:
            # because we use non-reversible msm, we want to silence the ImaginaryEigenvalueWarning
            import warnings
            from msmtools.util.exceptions import ImaginaryEigenValueWarning
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=ImaginaryEigenValueWarning,
                                        module='msmtools.analysis.dense.decomposition')
                correlation_time = max(1, msm_non_rev.timescales()[nstates - 1])
            # use the smaller of these two pessimistic estimates
            stride = int(min(lagtime, 2 * correlation_time))

        return stride
