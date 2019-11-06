
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

import numpy as np
from msmtools.dtraj import index_states

from sktime.base import Estimator
from sktime.markovprocess.hidden_markov_model import HMSM
from sktime.markovprocess.util import count_states


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
    def __init__(self, nstates=2, lag=1, stride=1, msm_init='largest-strong', reversible=True, stationary=False,
                 connectivity=None, mincount_connectivity='1/n', observe_nonempty=True, separate=None,
                 dt_traj='1 step', accuracy=1e-3, maxit=1000):
        super(MaximumLikelihoodHMSM, self).__init__()
        self.nstates = nstates
        self.lag = lag
        self.stride = stride
        self.msm_init = msm_init
        self.reversible = reversible
        self.stationary = stationary
        self.connectivity = connectivity
        if mincount_connectivity == '1/n':
            mincount_connectivity = 1.0/float(nstates)
        self.mincount_connectivity = mincount_connectivity
        self.separate = separate
        self.observe_nonempty = observe_nonempty
        self.dt_traj = dt_traj
        self.accuracy = accuracy
        self.maxit = maxit

    def _create_model(self) -> HMSM:
        return HMSM()

    def fit(self, dtrajs):
        from . import bhmm
        from .maximum_likelihood_msm import MaximumLikelihoodMSM
        from .bhmm.estimators.maximum_likelihood import MaximumLikelihoodEstimator
        import msmtools.estimation as msmest

        # CHECK LAG
        trajlengths = [np.size(dtraj) for dtraj in dtrajs]
        if self.lag >= np.max(trajlengths):
            raise ValueError(f'Illegal lag time {self.lag} exceeds longest trajectory length')
        if self.lag > np.mean(trajlengths):
            self.logger.warning(f'Lag time {self.lag} is on the order of mean trajectory length '
                                f'{np.mean(trajlengths)}. It is recommended to fit four lag times in each '
                                'trajectory. HMM might be inaccurate.')

        # EVALUATE STRIDE
        if self.stride == 'effective':
            # by default use lag as stride (=lag sampling), because we currently have no better theory for deciding
            # how many uncorrelated counts we can make
            self.stride = self.lag
            # get a quick fit from the spectral radius of the non-reversible
            msm_nr = MaximumLikelihoodMSM(lagtime=self.lag, reversible=False, sparse=False,
                                          dt_traj=self.dt_traj).fit(dtrajs).fetch_model()
            # if we have more than nstates timescales in our MSM, we use the next (neglected) timescale as an
            # fit of the decorrelation time
            if msm_nr.nstates > self.nstates:
                # because we use non-reversible msm, we want to silence the ImaginaryEigenvalueWarning
                import warnings
                from msmtools.util.exceptions import ImaginaryEigenValueWarning
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=ImaginaryEigenValueWarning,
                                            module='msmtools.analysis.dense.decomposition')
                    corrtime = max(1, msm_nr.timescales()[self.nstates - 1])
                # use the smaller of these two pessimistic estimates
                self.stride = int(min(self.lag, 2*corrtime))

        # LAG AND STRIDE DATA
        dtrajs_lagged_strided = bhmm.lag_observations(dtrajs, self.lag, stride=self.stride)

        # OBSERVATION SET
        if self.observe_nonempty:
            observe_subset = 'nonempty'
        else:
            observe_subset = None

        # INIT HMM
        from bhmm import init_discrete_hmm
        if self.msm_init=='largest-strong':
            hmm_init = init_discrete_hmm(dtrajs_lagged_strided, self.nstates, lag=1,
                                         reversible=self.reversible, stationary=True, regularize=True,
                                         method='lcs-spectral', separate=self.separate)
        elif self.msm_init=='all':
            hmm_init = init_discrete_hmm(dtrajs_lagged_strided, self.nstates, lag=1,
                                         reversible=self.reversible, stationary=True, regularize=True,
                                         method='spectral', separate=self.separate)
        elif isinstance(self.msm_init, (MaximumLikelihoodMSM)): #, OOMReweightedMSM)):  # initial MSM given.
            from bhmm.init.discrete import init_discrete_hmm_spectral
            p0, P0, pobs0 = init_discrete_hmm_spectral(self.msm_init.count_matrix_full, self.nstates,
                                                       reversible=self.reversible, stationary=True,
                                                       active_set=self.msm_init.active_set,
                                                       P=self.msm_init.transition_matrix, separate=self.separate)
            hmm_init = bhmm.discrete_hmm(p0, P0, pobs0)
            observe_subset = self.msm_init.active_set  # override observe_subset.
        else:
            raise ValueError(f'Unknown MSM initialization option: {self.msm_init}')

        # ---------------------------------------------------------------------------------------
        # Estimate discrete HMM
        # ---------------------------------------------------------------------------------------

        # run EM
        hmm_est = MaximumLikelihoodEstimator(dtrajs_lagged_strided, self.nstates, initial_model=hmm_init,
                                              output='discrete', reversible=self.reversible, stationary=self.stationary,
                                              accuracy=self.accuracy, maxit=self.maxit)
        # run
        hmm_est.fit()
        # package in discrete HMM
        self.hmm = bhmm.DiscreteHMM(hmm_est.hmm)

        # get model parameters
        self.initial_distribution = self.hmm.initial_distribution
        transition_matrix = self.hmm.transition_matrix
        observation_probabilities = self.hmm.output_probabilities

        # get estimation parameters
        self.likelihoods = hmm_est.likelihoods  # Likelihood history
        self.likelihood = self.likelihoods[-1]
        self.hidden_state_probabilities = hmm_est.hidden_state_probabilities  # gamma variables
        self.hidden_state_trajectories = hmm_est.hmm.hidden_state_trajectories  # Viterbi path
        self.count_matrix = hmm_est.count_matrix  # hidden count matrix
        self.initial_count = hmm_est.initial_count  # hidden init count
        self._active_set = np.arange(self.nstates)

        # TODO: it can happen that we loose states due to striding. Should we lift the output probabilities afterwards?
        # parametrize self
        self._dtrajs_full = dtrajs
        self._dtrajs_lagged = dtrajs_lagged_strided
        self._nstates_obs_full = msmest.number_of_states(dtrajs)
        self._nstates_obs = msmest.number_of_states(dtrajs_lagged_strided)
        self._observable_set = np.arange(self._nstates_obs)
        self._dtrajs_obs = dtrajs
        self.set_model_params(P=transition_matrix, pobs=observation_probabilities,
                              reversible=self.reversible, dt_model=self.timestep_traj.get_scaled(self.lag))

        # TODO: perhaps remove connectivity and just rely on .submodel()?
        # deal with connectivity
        states_subset = None
        if self.connectivity == 'largest':
            states_subset = 'largest-strong'
        elif self.connectivity == 'populous':
            states_subset = 'populous-strong'

        # return submodel (will return self if all None)
        return self.submodel(states=states_subset, obs=observe_subset,
                             mincount_connectivity=self.mincount_connectivity,
                             inplace=True)

    @property
    def msm_init(self):
        return self._msm_init

    @msm_init.setter
    def msm_init(self, value):
        from pyemma.msm.estimators import MaximumLikelihoodMSM
        from pyemma.msm.estimators import OOMReweightedMSM
        if (isinstance(value, (MaximumLikelihoodMSM, OOMReweightedMSM)) and not value._estimated):
            raise ValueError('Given initial msm has not been estimated. Input was {}'.format(value))
        self._msm_init = value

    @property
    def lagtime(self):
        """ The lag time in steps """
        return self.lag

    @property
    def nstates(self):
        return self._nstates

    @nstates.setter
    def nstates(self, value):
        # we override this setter here, because we want to avoid pyemma.msm.MSM class to overwrite our input parameter.
        # caused bug #1266
        if int(value) > 0 and value is not None:
            self._nstates = value

    @property
    def nstates_obs(self):
        r""" Number of states in discrete trajectories """
        return self._nstates_obs

    @property
    def active_set(self):
        """
        The active set of hidden states on which all hidden state computations are done

        """
        if hasattr(self, '_active_set'):
            return self._active_set
        else:
            return np.arange(self.nstates)  # all hidden states are active.

    @property
    def observable_set(self):
        """
        The active set of states on which all computations and estimations will be done

        """
        return self._observable_set

    @property
    def discrete_trajectories_lagged(self):
        """
        Transformed original trajectories that are used as an input into the HMM estimation

        """
        return self._dtrajs_lagged

    @property
    def discrete_trajectories_obs(self):
        """
        A list of integer arrays with the discrete trajectories mapped to the observation mode used.
        When using observe_active = True, the indexes will be given on the MSM active set. Frames that are not in the
        observation set will be -1. When observe_active = False, this attribute is identical to
        discrete_trajectories_full

        """
        return self._dtrajs_obs

    ################################################################################
    # Submodel functions using estimation information (counts)
    ################################################################################

    def submodel(self, states=None, obs=None, mincount_connectivity='1/n', inplace=False):
        """Returns a HMM with restricted state space

        Parameters
        ----------
        states : None, str or int-array
            Hidden states to restrict the model to. In addition to specifying
            the subset, possible options are:
            * None : all states - don't restrict
            * 'populous-strong' : strongly connected subset with maximum counts
            * 'populous-weak' : weakly connected subset with maximum counts
            * 'largest-strong' : strongly connected subset with maximum size
            * 'largest-weak' : weakly connected subset with maximum size
        obs : None, str or int-array
            Observed states to restrict the model to. In addition to specifying
            an array with the state labels to be observed, possible options are:
            * None : all states - don't restrict
            * 'nonempty' : all states with at least one observation in the estimator
        mincount_connectivity : float or '1/n'
            minimum number of counts to consider a connection between two states.
            Counts lower than that will count zero in the connectivity check and
            may thus separate the resulting transition matrix. Default value:
            1/nstates.
        inplace : Bool
            if True, submodel is estimated in-place, overwriting the original
            estimator and possibly discarding information. Default value: False

        Returns
        -------
        hmm : HMM
            The restricted HMM.

        """
        if states is None and obs is None and mincount_connectivity == 0:
            return self
        if states is None:
            states = np.arange(self.nstates)
        if obs is None:
            obs = np.arange(self.nstates_obs)

        if str(mincount_connectivity) == '1/n':
            mincount_connectivity = 1.0/float(self.nstates)

        # handle new connectivity
        from bhmm.estimators import _tmatrix_disconnected
        S = _tmatrix_disconnected.connected_sets(self.count_matrix,
                                                 mincount_connectivity=mincount_connectivity,
                                                 strong=True)
        if inplace:
            submodel_estimator = self
        else:
            from copy import deepcopy
            submodel_estimator = deepcopy(self)

        if len(S) > 1:
            # keep only non-negligible transitions
            C = np.zeros(self.count_matrix.shape)
            large = np.where(self.count_matrix >= mincount_connectivity)
            C[large] = self.count_matrix[large]
            for s in S:  # keep all (also small) transition counts within strongly connected subsets
                C[np.ix_(s, s)] = self.count_matrix[np.ix_(s, s)]
            # re-fit transition matrix with disc.
            P = _tmatrix_disconnected.estimate_P(C, reversible=self.reversible, mincount_connectivity=0)
            pi = _tmatrix_disconnected.stationary_distribution(P, C)
        else:
            C = self.count_matrix
            P = self.transition_matrix
            pi = self.stationary_distribution

        # determine substates
        if isinstance(states, str):
            from bhmm.estimators import _tmatrix_disconnected
            strong = 'strong' in states
            largest = 'largest' in states
            S = _tmatrix_disconnected.connected_sets(self.count_matrix, mincount_connectivity=mincount_connectivity,
                                                     strong=strong)
            if largest:
                score = [len(s) for s in S]
            else:
                score = [self.count_matrix[np.ix_(s, s)].sum() for s in S]
            states = np.array(S[np.argmax(score)])
        if states is not None:  # sub-transition matrix
            submodel_estimator._active_set = states
            C = C[np.ix_(states, states)].copy()
            P = P[np.ix_(states, states)].copy()
            P /= P.sum(axis=1)[:, None]
            pi = _tmatrix_disconnected.stationary_distribution(P, C)
            submodel_estimator.initial_count = self.initial_count[states]
            submodel_estimator.initial_distribution = self.initial_distribution[states] / self.initial_distribution[states].sum()

        # determine observed states
        if str(obs) == 'nonempty':
            obs = np.where(count_states(self.discrete_trajectories_lagged) > 0)[0]
        if obs is not None:
            # set observable set
            submodel_estimator._observable_set = obs
            submodel_estimator._nstates_obs = obs.size
            # full2active mapping
            _full2obs = -1 * np.ones(self._nstates_obs_full, dtype=int)
            _full2obs[obs] = np.arange(len(obs), dtype=int)
            # observable trajectories
            submodel_estimator._dtrajs_obs = []
            for dtraj in self.discrete_trajectories_full:
                submodel_estimator._dtrajs_obs.append(_full2obs[dtraj])

            # observation matrix
            B = self.observation_probabilities[np.ix_(states, obs)].copy()
            B /= B.sum(axis=1)[:, None]
        else:
            B = self.observation_probabilities

        # set quantities back.
        submodel_estimator.update_model_params(P=P, pobs=B, pi=pi)
        submodel_estimator.count_matrix_EM = self.count_matrix[np.ix_(states, states)]  # unchanged count matrix
        submodel_estimator.count_matrix = C  # count matrix consistent with P
        return submodel_estimator

    def submodel_largest(self, strong=True, mincount_connectivity='1/n'):
        """ Returns the largest connected sub-HMM (convenience function)

        Returns
        -------
        hmm : HMM
            The restricted HMM.

        """
        if strong:
            return self.submodel(states='largest-strong', mincount_connectivity=mincount_connectivity)
        else:
            return self.submodel(states='largest-weak', mincount_connectivity=mincount_connectivity)

    def submodel_populous(self, strong=True, mincount_connectivity='1/n'):
        """ Returns the most populous connected sub-HMM (convenience function)

        Returns
        -------
        hmm : HMM
            The restricted HMM.

        """
        if strong:
            return self.submodel(states='populous-strong', mincount_connectivity=mincount_connectivity)
        else:
            return self.submodel(states='populous-weak', mincount_connectivity=mincount_connectivity)

    def submodel_disconnect(self, mincount_connectivity='1/n'):
        """Disconnects sets of hidden states that are barely connected

        Runs a connectivity check excluding all transition counts below
        mincount_connectivity. The transition matrix and stationary distribution
        will be re-estimated. Note that the resulting transition matrix
        may have both strongly and weakly connected subsets.

        Parameters
        ----------
        mincount_connectivity : float or '1/n'
            minimum number of counts to consider a connection between two states.
            Counts lower than that will count zero in the connectivity check and
            may thus separate the resulting transition matrix. The default
            evaluates to 1/nstates.

        Returns
        -------
        hmm : HMM
            The restricted HMM.

        """
        return self.submodel(mincount_connectivity=mincount_connectivity)

    def trajectory_weights(self):
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
        hist = 1.0 * msmtraj.count_states(self.discrete_trajectories_obs, ignore_negative=True)
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

    @property
    def observable_state_indexes(self):
        """
        Ensures that the observable states are indexed and returns the indices
        """
        try:  # if we have this attribute, return it
            return self._observable_state_indexes
        except AttributeError:  # didn't exist? then create it.
            self._observable_state_indexes = index_states(self.discrete_trajectories_obs)
            return self._observable_state_indexes

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
        import pyemma.util.discrete_trajectories as dt
        return dt.sample_indexes_by_distribution(self.observable_state_indexes, self.observation_probabilities, nsample)


################################################################################
# Model Validation
################################################################################

def cktest(self, mlags=10, conf=0.95, err_est=False, n_jobs=None, show_progress=True):
    """ Conducts a Chapman-Kolmogorow test.

    Parameters
    ----------
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
    ck = ChapmanKolmogorovValidator(self, self, np.eye(self.nstates),
                                    mlags=mlags, conf=conf, err_est=err_est)
    if data is not None:
        ck.fit(dtrajs)
    return ck
