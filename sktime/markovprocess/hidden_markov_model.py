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
import typing
from msmtools.dtraj import index_states

from sktime.markovprocess import MarkovStateModel, transition_counting
from sktime.markovprocess.bhmm import lag_observations
from sktime.markovprocess.bhmm.estimators import _tmatrix_disconnected
from sktime.markovprocess.util import count_states
from sktime.numeric import mdot
from sktime.util import ensure_ndarray, ensure_dtraj_list

from sktime.markovprocess.bhmm.hmm.generic_hmm import HMM as BHMM_HMM


class HMMTransitionCountModel(transition_counting.TransitionCountModel):
    def __init__(self, n_states=None, observable_set: typing.Optional[np.ndarray]=None,
                 stride=1, initial_count=None, symbols=None,
                 lagtime=1, active_set=None, dt_traj='1 step',
                 connected_sets=(), count_matrix=None, state_histogram=None):
        super(HMMTransitionCountModel, self).__init__(lagtime=lagtime, active_set=active_set, dt_traj=dt_traj,
                                                      connected_sets=connected_sets, count_matrix=count_matrix,
                                                      state_histogram=state_histogram)

        self.initial_count = initial_count
        self._n_states_full = n_states
        self._observable_set = observable_set
        self._n_states_obs = observable_set.size
        self._stride = stride
        self._symbols = symbols

    @property
    def stride(self):
        """ Stride with which the dtrajs were lagged and stridden """
        return self._stride

    @property
    def symbols(self):
        """Sorted unique symbols in observations """
        return self._symbols

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
        return super(HMMTransitionCountModel, self).count_matrix

    @property
    def n_states_obs(self):
        return self._n_states_obs

    @property
    def observable_set(self):
        return self._observable_set

    @staticmethod
    def compute_dtrajs_effective(dtrajs, lagtime, n_states, stride):
        lagtime = int(lagtime)
        # EVALUATE STRIDE
        if stride == 'effective':
            from sktime.markovprocess.util import compute_effective_stride
            stride = compute_effective_stride(dtrajs, lagtime, n_states)

        # LAG AND STRIDE DATA
        dtrajs_lagged_strided = lag_observations(dtrajs, lagtime, stride=stride)
        return dtrajs_lagged_strided


class HMSM(MarkovStateModel):
    r""" Hidden Markov model on discrete states.

    Parameters
    ----------
    transition_matrix : ndarray (m,m)
        coarse-grained or hidden transition matrix

    p_obs : ndarray (m,n)
        observation probability matrix from hidden to observable discrete states

    pi: ndarray(m), optional
        stationary distribution

    dt_model : str, optional, default='1 step'
        time step of the model

    neig:

    reversible:

    initial_distribution:

    """

    def __init__(self, transition_matrix, observation_probabilities, pi=None, dt_model='1 step',
                 neig=None, reversible=None, count_model=None, initial_distribution=None, bhmm_model : BHMM_HMM = None):
        if count_model is None:
            count_model = HMMTransitionCountModel()

        # construct superclass and check input
        super(HMSM, self).__init__(transition_matrix=transition_matrix, pi=pi, dt_model=dt_model,
                                   reversible=reversible, neig=neig, count_model=count_model)

        # assert types.is_float_matrix(pobs), 'pobs is not a matrix of floating numbers'
        observation_probabilities = ensure_ndarray(observation_probabilities, ndim=2, dtype=np.float64)
        assert np.allclose(observation_probabilities.sum(axis=1), 1), 'pobs is not a stochastic matrix'
        self._n_states_obs = observation_probabilities.shape[1]
        self._observation_probabilities = observation_probabilities
        self._initial_distribution = initial_distribution
        self._hmm = bhmm_model

    @property
    def bhmm_model(self) -> BHMM_HMM:
        return self._hmm

    @property
    def count_model(self) -> typing.Optional[HMMTransitionCountModel]:
        return self._count_model

    @count_model.setter
    def count_model(self, value: typing.Optional[HMMTransitionCountModel]):
        self.count_model = value

    ################################################################################
    # Submodel functions using estimation information (counts)
    ################################################################################
    def submodel(self, states: typing.Optional[np.ndarray] = None, obs: typing.Optional[np.ndarray] = None,
                 mincount_connectivity='1/n'):
        """Returns a HMM with restricted state space

        Parameters
        ----------
        states : None or int-array
            Hidden states to restrict the model to. In addition to specifying
            the subset, possible options are:

            * int-array: indices of states to restrict onto
            * None : all states - don't restrict

        obs : None or int-array
            Observed states to restrict the model to. In addition to specifying
            an array with the state labels to be observed, possible options are:

              * int-array: indices of states to restrict onto
              * None : all states - don't restrict

        mincount_connectivity : float or '1/n'
            minimum number of counts to consider a connection between two states.
            Counts lower than that will count zero in the connectivity check and
            may thus separate the resulting transition matrix. Default value:
            1/n_states.
        inplace : Bool
            if True, submodel is estimated in-place, overwriting the original
            estimator and possibly discarding information. Default value: False

        Returns
        -------
        hmm : HMM
            The restricted HMM.
        """
        if self.count_model is None:
            return self._submodel(states=states, obs=obs)

        if states is None:
            states = np.arange(self.n_states)
        if obs is None:
            obs = np.arange(self.n_states_obs)

        count_matrix = self.count_model.count_matrix.copy()
        assert count_matrix is not None

        if str(mincount_connectivity) == '1/n':
            mincount_connectivity = 1.0 / float(self.n_states)

        # handle new connectivity
        from sktime.markovprocess.bhmm.estimators import _tmatrix_disconnected
        S = _tmatrix_disconnected.connected_sets(count_matrix,
                                                 mincount_connectivity=mincount_connectivity,
                                                 strong=True)

        if len(S) > 1:
            # keep only non-negligible transitions
            C = np.zeros(count_matrix.shape)
            large = np.where(count_matrix >= mincount_connectivity)
            C[large] = count_matrix[large]
            for s in S:  # keep all (also small) transition counts within strongly connected subsets
                C[np.ix_(s, s)] = count_matrix[np.ix_(s, s)]
            # re-fit transition matrix with disc.
            P = _tmatrix_disconnected.estimate_P(C, reversible=self.is_reversible, mincount_connectivity=0)
        else:
            C = count_matrix
            P = self.transition_matrix.copy()

        # sub-transition matrix
        P = P[np.ix_(states, states)]
        P /= P.sum(axis=1)[:, None]
        C = C[np.ix_(states, states)]
        pi = _tmatrix_disconnected.stationary_distribution(P, C)
        initial_count = self.count_model.initial_count[states].copy()
        initial_distribution = self.initial_distribution[states] / self.initial_distribution[states].sum()

        # # full2active mapping
        # _full2obs = -1 * np.ones(se   lf._n_states_obs_full, dtype=int)
        # _full2obs[obs] = np.arange(len(obs), dtype=int)
        # # observable trajectories
        # model._dtrajs_obs = []
        # for dtraj in self.count_model.discrete_trajectories_full:
        #     model._dtrajs_obs.append(_full2obs[dtraj])

        # observation matrix
        B = self.observation_probabilities[np.ix_(states, obs)].copy()
        B /= B.sum(axis=1)[:, None]

        count_model = HMMTransitionCountModel(
            n_states=self.count_model.n_states_full, observable_set=obs,
            stride=self.count_model.stride, symbols=self.count_model.symbols, dt_traj=self.count_model.dt_traj,
            state_histogram=self.count_model.state_histogram,
            initial_count=initial_count, active_set=states,
            connected_sets=S, count_matrix=C,
        )
        model = HMSM(transition_matrix=P, observation_probabilities=B, pi=pi, dt_model=self.dt_model, neig=self.neig,
                     reversible=self.is_reversible, count_model=count_model,
                     initial_distribution=initial_distribution, bhmm_model=self.bhmm_model)
        return model

    def _select_states(self, mincount_connectivity, states):
        if str(mincount_connectivity) == '1/n':
            mincount_connectivity = 1.0 / float(self.n_states)
        if isinstance(states, str):
            strong = 'strong' in states
            largest = 'largest' in states
            S = _tmatrix_disconnected.connected_sets(self.count_model.count_matrix,
                                                     mincount_connectivity=mincount_connectivity,
                                                     strong=strong)
            if largest:
                score = [len(s) for s in S]
            else:
                score = [self.count_model.count_matrix[np.ix_(s, s)].sum() for s in S]
            states = np.array(S[np.argmax(score)])
        return states

    def nonempty_obs(self, dtrajs):
        if dtrajs is None:
            raise ValueError("Needs nonempty dtrajs to evaluate nonempty obs.")
        dtrajs = ensure_dtraj_list(dtrajs)
        dtrajs_lagged_strided = self.count_model.compute_dtrajs_effective(
            dtrajs, self.count_model.lagtime, self.count_model.n_states_full, self.count_model.stride
        )
        obs = np.where(count_states(dtrajs_lagged_strided) > 0)[0]
        return obs

    def states_largest(self, strong=True, mincount_connectivity='1/n'):
        return self._select_states(mincount_connectivity, 'largest-strong' if strong else 'largest-weak')

    def submodel_largest(self, strong=True, mincount_connectivity='1/n', observe_nonempty=True, dtrajs=None):
        """ Returns the largest connected sub-HMM (convenience function)

        Returns
        -------
        hmm : HMSM
            The restricted HMSM.

        """
        states = self.states_largest(strong=strong, mincount_connectivity=mincount_connectivity)
        obs = self.nonempty_obs(dtrajs) if observe_nonempty else None
        return self.submodel(states=states, obs=obs, mincount_connectivity=mincount_connectivity)

    def states_populous(self, strong=True, mincount_connectivity='1/n'):
        return self._select_states(mincount_connectivity, 'populous-strong' if strong else 'populous-weak')

    def submodel_populous(self, strong=True, mincount_connectivity='1/n', observe_nonempty=True, dtrajs=None):
        """ Returns the most populous connected sub-HMM (convenience function)

        Returns
        -------
        hmm : HMSM
            The restricted HMSM.

        """
        states = self.states_populous(strong=strong, mincount_connectivity=mincount_connectivity)
        obs = self.nonempty_obs(dtrajs) if observe_nonempty else None
        return self.submodel(states=states, obs=obs, mincount_connectivity=mincount_connectivity)

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
            evaluates to 1/n_states.

        Returns
        -------
        hmm : HMSM
            The restricted HMM.

        """
        return self.submodel(mincount_connectivity=mincount_connectivity)

    @property
    def observation_probabilities(self):
        r""" returns the output probability matrix

        Returns
        -------
        Pout : ndarray (m,n)
            output probability matrix from hidden to observable discrete states

        """
        return self._observation_probabilities

    @property
    def n_states_obs(self):
        return self.observation_probabilities.shape[1]

    @property
    def initial_distribution(self):
        return self._initial_distribution

    @property
    def lifetimes(self):
        r""" Lifetimes of states of the hidden transition matrix

        Returns
        -------
        l : ndarray(n_states)
            state lifetimes in units of the input trajectory time step,
            defined by :math:`-\tau / ln \mid p_{ii} \mid, i = 1,...,n_states`, where
            :math:`p_{ii}` are the diagonal entries of the hidden transition matrix.

        """
        return -self._dt_model / np.log(np.diag(self.transition_matrix))

    def transition_matrix_obs(self, k=1):
        r""" Computes the transition matrix between observed states

        Transition matrices for longer lag times than the one used to
        parametrize this HMSM can be obtained by setting the k option.
        Note that a HMSM is not Markovian, thus we cannot compute transition
        matrices at longer lag times using the Chapman-Kolmogorow equality.
        I.e.:

        .. math::
            P (k \tau) \neq P^k (\tau)

        This function computes the correct transition matrix using the
        metastable (coarse) transition matrix :math:`P_c` as:

        .. math::
            P (k \tau) = {\Pi}^-1 \chi^{\top} ({\Pi}_c) P_c^k (\tau) \chi

        where :math:`\chi` is the output probability matrix, :math:`\Pi_c` is
        a diagonal matrix with the metastable-state (coarse) stationary
        distribution and :math:`\Pi` is a diagonal matrix with the
        observable-state stationary distribution.

        Parameters
        ----------
        k : int, optional, default=1
            Multiple of the lag time for which the
            By default (k=1), the transition matrix at the lag time used to
            construct this HMSM will be returned. If a higher power is given,

        """
        Pi_c = np.diag(self.stationary_distribution)
        P_c = self.transition_matrix
        P_c_k = np.linalg.matrix_power(P_c, k)  # take a power if needed
        B = self.observation_probabilities
        C = np.dot(np.dot(B.T, Pi_c), np.dot(P_c_k, B))
        P = C / C.sum(axis=1)[:, None]  # row normalization
        return P

    @property
    def stationary_distribution_obs(self):
        return np.dot(self.stationary_distribution, self.observation_probabilities)

    @property
    def eigenvectors_left_obs(self):
        return np.dot(self.eigenvectors_left(), self.observation_probabilities)

    @property
    def eigenvectors_right_obs(self):
        return np.dot(self.metastable_memberships, self.eigenvectors_right())

    def propagate(self, p0, k):
        r""" Propagates the initial distribution p0 k times

        Computes the product

        .. math::

            p_k = p_0^T P^k

        If the lag time of transition matrix :math:`P` is :math:`\tau`, this
        will provide the probability distribution at time :math:`k \tau`.

        Parameters
        ----------
        p0 : ndarray(n)
            Initial distribution. Vector of size of the active set.

        k : int
            Number of time steps

        Returns
        ----------
        pk : ndarray(n)
            Distribution after k steps. Vector of size of the active set.

        """
        # p0 = types.ensure_ndarray(p0, ndim=1, kind='numeric')
        # assert types.is_int(k) and k >= 0, 'k must be a non-negative integer'
        if k == 0:  # simply return p0 normalized
            return p0 / p0.sum()

        micro = False
        # are we on microstates space?
        if len(p0) == self.n_states_obs:
            micro = True
            # project to hidden and compute
            p0 = np.dot(self.observation_probabilities, p0)

        self._ensure_eigendecomposition(self.n_states)
        # TODO: eigenvectors_right() and so forth call ensure_eigendecomp again with self.neig instead of self.n_states
        pk = mdot(p0.T, self.eigenvectors_right(), np.diag(np.power(self.eigenvalues(), k)), self.eigenvectors_left())

        if micro:
            pk = np.dot(pk, self.observation_probabilities)  # convert back to microstate space

        # normalize to 1.0 and return
        return pk / pk.sum()

    def _submodel(self, states=None, obs=None):
        """Returns a HMM with restricted state space (only restrict states and observations, not counts

        Parameters
        ----------
        states : None or int-array
            Hidden states to restrict the model to (if not None).
        obs : None, str or int-array
            Observed states to restrict the model to (if not None).

        Returns
        -------
        hmm : HMM
            The restricted HMM.

        """
        assert self.count_model is None

        if states is None and obs is None:
            return self  # do nothing
        if states is None:
            states = np.arange(self.n_states)
        if obs is None:
            obs = np.arange(self.n_states_obs)

        # transition matrix
        P = self.transition_matrix[np.ix_(states, states)].copy()
        P /= P.sum(axis=1)[:, None]

        # observation matrix
        B = self.observation_probabilities[np.ix_(states, obs)].copy()
        B /= B.sum(axis=1)[:, None]

        return HMSM(P, B, dt_model=self.dt_model, reversible=self.is_reversible)

    # ================================================================================================================
    # Experimental properties: Here we allow to use either coarse-grained or microstate observables
    # ================================================================================================================

    def expectation(self, a):
        a = ensure_ndarray(a, dtype=np.float64)
        # are we on microstates space?
        if len(a) == self.n_states_obs:
            # project to hidden and compute
            a = np.dot(self.observation_probabilities, a)
        # now we are on macrostate space, or something is wrong
        if len(a) == self.n_states:
            return super(HMSM, self).expectation(a)
        else:
            raise ValueError(
                f'observable vectors have size {len(a)} which is incompatible with both hidden ({self.n_states})'
                f' and observed states ({self.n_states_obs})')

    def correlation(self, a, b=None, maxtime=None, k=None, ncv=None):
        # basic checks for a and b
        a = ensure_ndarray(a, ndim=1)
        b = ensure_ndarray(b, ndim=1, size=len(a), allow_None=True)
        # are we on microstates space?
        if len(a) == self.n_states_obs:
            a = np.dot(self.observation_probabilities, a)
            if b is not None:
                b = np.dot(self.observation_probabilities, b)
        # now we are on macrostate space, or something is wrong
        if len(a) == self.n_states:
            return super(HMSM, self).correlation(a, b=b, maxtime=maxtime)
        else:
            raise ValueError(
                f'observable vectors have size {len(a)} which is incompatible with both hidden ({self.n_states})'
                f' and observed states ({self.n_states_obs})')

    def fingerprint_correlation(self, a, b=None, k=None, ncv=None):
        # basic checks for a and b
        a = ensure_ndarray(a, ndim=1)
        b = ensure_ndarray(b, ndim=1, size=len(a), allow_None=True)
        # are we on microstates space?
        if len(a) == self.n_states_obs:
            a = np.dot(self.observation_probabilities, a)
            if b is not None:
                b = np.dot(self.observation_probabilities, b)
        # now we are on macrostate space, or something is wrong
        if len(a) == self.n_states:
            return super(HMSM, self).fingerprint_correlation(a, b=b)
        else:
            raise ValueError(
                f'observable vectors have size {len(a)} which is incompatible with both hidden ({self.n_states})'
                f' and observed states ({self.n_states_obs})')

    def relaxation(self, p0, a, maxtime=None, k=None, ncv=None):
        # basic checks for a and b
        p0 = ensure_ndarray(p0, ndim=1)
        a = ensure_ndarray(a, ndim=1, size=len(p0))
        # are we on microstates space?
        if len(a) == self.n_states_obs:
            p0 = np.dot(self.observation_probabilities, p0)
            a = np.dot(self.observation_probabilities, a)
        # now we are on macrostate space, or something is wrong
        if len(a) == self.n_states:
            return super(HMSM, self).relaxation(p0, a, maxtime=maxtime)
        else:
            raise ValueError(
                f'observable vectors have size {len(a)} which is incompatible with both hidden ({self.n_states})'
                f' and observed states ({self.n_states_obs})')

    def fingerprint_relaxation(self, p0, a, k=None, ncv=None):
        # basic checks for a and b
        p0 = ensure_ndarray(p0, ndim=1)
        a = ensure_ndarray(a, ndim=1, size=len(p0))
        # are we on microstates space?
        if len(a) == self.n_states_obs:
            p0 = np.dot(self.observation_probabilities, p0)
            a = np.dot(self.observation_probabilities, a)
        # now we are on macrostate space, or something is wrong
        if len(a) == self.n_states:
            return super(HMSM, self).fingerprint_relaxation(p0, a)
        else:
            raise ValueError('observable vectors have size %s which is incompatible with both hidden (%s)'
                             ' and observed states (%s)' % (len(a), self.n_states, self.n_states_obs))

    def pcca(self, m):
        raise NotImplementedError('PCCA is not meaningful for Hidden Markov models. '
                                  'If you really want to do this, initialize an MSM with the HMSM transition matrix.')

    # ================================================================================================================
    # Metastable state stuff is overwritten, because we now have the HMM output probability matrix
    # ================================================================================================================

    @property
    def metastable_memberships(self):
        r""" Computes the memberships of observable states to metastable sets by
            Bayesian inversion as described in [1]_.

        Returns
        -------
        M : ndarray((n,m))
            A matrix containing the probability or membership of each
            observable state to be assigned to each metastable or hidden state.
            The row sums of M are 1.

        References
        ----------
        .. [1] F. Noe, H. Wu, J.-H. Prinz and N. Plattner: Projected and hidden
            Markov models for calculating kinetics and metastable states of
            complex molecules. J. Chem. Phys. 139, 184114 (2013)

        """
        nonzero = np.nonzero(self.stationary_distribution_obs)[0]
        M = np.zeros((self.n_states_obs, self.n_states))
        M[nonzero, :] = np.transpose(np.diag(self.stationary_distribution).dot(
            self.observation_probabilities[:, nonzero]) / self.stationary_distribution_obs[nonzero])
        # renormalize
        M[nonzero, :] /= M.sum(axis=1)[nonzero, None]
        return M

    @property
    def metastable_distributions(self):
        r""" Returns the output probability distributions. Identical to
            :meth:`observation_probabilities`

        Returns
        -------
        Pout : ndarray (m,n)
            output probability matrix from hidden to observable discrete states

        See also
        --------
        observation_probabilities

        """
        return self.observation_probabilities

    @property
    def metastable_sets(self):
        r""" Computes the metastable sets of observable states within each
            metastable set

        Notes
        -----
        This is only recommended for visualization purposes. You *cannot*
        compute any actual quantity of the coarse-grained kinetics without
        employing the fuzzy memberships!

        Returns
        -------
        sets : list of int-arrays
            A list of length equal to metastable states. Each element is an array
            with observable state indexes contained in it

        """
        res = []
        assignment = self.metastable_assignments
        for i in range(self.n_states):
            res.append(np.where(assignment == i)[0])
        return res

    @property
    def metastable_assignments(self):
        r""" Computes the assignment to metastable sets for observable states

        Notes
        -----
        This is only recommended for visualization purposes. You *cannot*
        compute any actual quantity of the coarse-grained kinetics without
        employing the fuzzy memberships!

        Returns
        -------
        ndarray((n) ,dtype=int)
            For each observable state, the metastable state it is located in.

        """
        return np.argmax(self.observation_probabilities, axis=0)

    def simulate(self, N, start=None, stop=None, dt=1):
        """
        Generates a realization of the Hidden Markov Model

        Parameters
        ----------
        N : int
            trajectory length in steps of the lag time
        start : int, optional, default = None
            starting hidden state. If not given, will sample from the stationary
            distribution of the hidden transition matrix.
        stop : int or int-array-like, optional, default = None
            stopping hidden set. If given, the trajectory will be stopped before
            N steps once a hidden state of the stop set is reached
        dt : int
            trajectory will be saved every dt time steps.
            Internally, the dt'th power of P is taken to ensure a more efficient simulation.

        Returns
        -------
        htraj : (N/dt, ) ndarray
            The hidden state trajectory with length N/dt
        otraj : (N/dt, ) ndarray
            The observable state discrete trajectory with length N/dt

        """

        from scipy import stats
        import msmtools.generation as msmgen
        # generate output distributions
        # TODO: replace this with numpy.random.choice
        output_distributions = [stats.rv_discrete(values=(np.arange(self._observation_probabilities.shape[1]), pobs_i)) for pobs_i in
                                self._observation_probabilities]
        # sample hidden trajectory
        htraj = msmgen.generate_traj(self.transition_matrix, N, start=start, stop=stop, dt=dt)
        otraj = np.zeros(htraj.size, dtype=int)
        # for each time step, sample microstate
        for t, h in enumerate(htraj):
            otraj[t] = output_distributions[h].rvs()  # current cluster
        return htraj, otraj

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
