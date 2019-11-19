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

from sktime.markovprocess import MarkovStateModel
from sktime.markovprocess.util import count_states
from sktime.numeric import mdot
from sktime.util import ensure_ndarray


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

    """

    def __init__(self, transition_matrix=None, pobs=None, pi=None, dt_model='1 step',
                 neig=None, reversible=None, count_model=None, initial_distribution=None):
        if count_model is None:
            from sktime.markovprocess.maximum_likelihood_hmsm import _HMMTransitionCounts
            count_model = _HMMTransitionCounts()

        # construct superclass and check input
        super(HMSM, self).__init__(transition_matrix=transition_matrix, pi=pi, dt_model=dt_model,
                                   reversible=reversible, neig=neig, count_model=count_model)

        if pobs is not None:
            #assert types.is_float_matrix(pobs), 'pobs is not a matrix of floating numbers'
            pobs = ensure_ndarray(pobs, ndim=2, dtype=np.float64)
            assert np.allclose(pobs.sum(axis=1), 1), 'pobs is not a stochastic matrix'
            self._nstates_obs = pobs.shape[1]
            # TODO: refactor
            self.pobs = pobs
        self._initial_distribution = initial_distribution

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
        if self.count_model is None:
            return self._submodel(states=states, obs=obs)

        if states is None and obs is None and mincount_connectivity == 0:
            return self
        if states is None:
            states = np.arange(self.nstates)
        if obs is None:
            obs = np.arange(self.nstates_obs)

        if str(mincount_connectivity) == '1/n':
            mincount_connectivity = 1.0/float(self.nstates)

        # should we always take a copy here?
        model = self.copy() if not inplace else self

        count_matrix = model.count_model.count_matrix
        assert count_matrix is not None

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
            P = _tmatrix_disconnected.estimate_P(C, reversible=self.reversible, mincount_connectivity=0)
            pi = _tmatrix_disconnected.stationary_distribution(P, C)
        else:
            C = count_matrix
            P = self.transition_matrix
            pi = self.stationary_distribution

        # determine substates
        if isinstance(states, str):
            strong = 'strong' in states
            largest = 'largest' in states
            S = _tmatrix_disconnected.connected_sets(count_matrix,
                                                     mincount_connectivity=mincount_connectivity,
                                                     strong=strong)
            if largest:
                score = [len(s) for s in S]
            else:
                score = [count_matrix[np.ix_(s, s)].sum() for s in S]
            states = np.array(S[np.argmax(score)])
        if states is not None:  # sub-transition matrix
            model._active_set = states
            C = C[np.ix_(states, states)].copy()
            P = P[np.ix_(states, states)].copy()
            P /= P.sum(axis=1)[:, None]
            pi = _tmatrix_disconnected.stationary_distribution(P, C)
            model.count_model.initial_count = self.count_model.initial_count[states]
            model._initial_distribution = self.initial_distribution[states] / self.initial_distribution[states].sum()

        # determine observed states
        if str(obs) == 'nonempty':
            obs = np.where(count_states(self.count_model.dtrajs_lagged_strided) > 0)[0]
        if obs is not None:
            pass

            # TODO: are these count_model attributes?
            # set observable set
            # model._observable_set = obs
            # model._nstates_obs = obs.size
            # # full2active mapping
            # _full2obs = -1 * np.ones(se   lf._nstates_obs_full, dtype=int)
            # _full2obs[obs] = np.arange(len(obs), dtype=int)
            # # observable trajectories
            # model._dtrajs_obs = []
            # for dtraj in self.count_model.discrete_trajectories_full:
            #     model._dtrajs_obs.append(_full2obs[dtraj])

            # observation matrix
            B = self.observation_probabilities[np.ix_(states, obs)].copy()
            B /= B.sum(axis=1)[:, None]
        else:
            B = self.observation_probabilities

        # set quantities back.
        cm = model.count_model
        cm._count_matrix_EM = cm.count_matrix[np.ix_(states, states)]  # unchanged count matrix
        model.__init__(transition_matrix=P, pobs=B, pi=pi,
                       dt_model=model.dt_model,
                       reversible=model.reversible,
                       count_model=model.count_model,
                       initial_distribution=model.initial_distribution)
        return model

    def submodel_largest(self, strong=True, mincount_connectivity='1/n'):
        """ Returns the largest connected sub-HMM (convenience function)

        Returns
        -------
        hmm : HMSM
            The restricted HMSM.

        """
        if strong:
            return self.submodel(states='largest-strong', mincount_connectivity=mincount_connectivity)
        else:
            return self.submodel(states='largest-weak', mincount_connectivity=mincount_connectivity)

    def submodel_populous(self, strong=True, mincount_connectivity='1/n'):
        """ Returns the most populous connected sub-HMM (convenience function)

        Returns
        -------
        hmm : HMSM
            The restricted HMSM.

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
        return self.pobs

    @property
    def nstates_obs(self):
        return self.observation_probabilities.shape[1]

    @property
    def initial_distribution(self):
        return self._initial_distribution

    @property
    def lifetimes(self):
        r""" Lifetimes of states of the hidden transition matrix

        Returns
        -------
        l : ndarray(nstates)
            state lifetimes in units of the input trajectory time step,
            defined by :math:`-\tau / ln \mid p_{ii} \mid, i = 1,...,nstates`, where
            :math:`p_{ii}` are the diagonal entries of the hidden transition matrix.

        """
        return -self._timeunit_model.dt / np.log(np.diag(self.transition_matrix))

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
        #p0 = types.ensure_ndarray(p0, ndim=1, kind='numeric')
        #assert types.is_int(k) and k >= 0, 'k must be a non-negative integer'
        if k == 0:  # simply return p0 normalized
            return p0 / p0.sum()

        micro = False
        # are we on microstates space?
        if len(p0) == self.nstates_obs:
            micro = True
            # project to hidden and compute
            p0 = np.dot(self.observation_probabilities, p0)

        self._ensure_eigendecomposition(self.nstates)
        # TODO: eigenvectors_right() and so forth call ensure_eigendecomp again with self.neig instead of self.nstates
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
            states = np.arange(self.nstates)
        if obs is None:
            obs = np.arange(self.nstates_obs)

        # transition matrix
        P = self.transition_matrix[np.ix_(states, states)].copy()
        P /= P.sum(axis=1)[:, None]

        # observation matrix
        B = self.observation_probabilities[np.ix_(states, obs)].copy()
        B /= B.sum(axis=1)[:, None]

        sub_hmsm = HMSM(P, B, dt_model=self.dt_model)
        #TODO: shouldn't this be true in every case?
        sub_hmsm.reversible = self.reversible
        return sub_hmsm

    # ================================================================================================================
    # Experimental properties: Here we allow to use either coarse-grained or microstate observables
    # ================================================================================================================

    def expectation(self, a):
        a = ensure_ndarray(a, dtype=np.float64)
        # are we on microstates space?
        if len(a) == self.nstates_obs:
            # project to hidden and compute
            a = np.dot(self.observation_probabilities, a)
        # now we are on macrostate space, or something is wrong
        if len(a) == self.nstates:
            return super(HMSM, self).expectation(a)
        else:
            raise ValueError(f'observable vectors have size {len(a)} which is incompatible with both hidden ({self.nstates})'
                             f' and observed states ({self.nstates_obs})')

    def correlation(self, a, b=None, maxtime=None, k=None, ncv=None):
        # basic checks for a and b
        a = ensure_ndarray(a, ndim=1)
        b = ensure_ndarray(b, ndim=1, size=len(a), allow_None=True)
        # are we on microstates space?
        if len(a) == self.nstates_obs:
            a = np.dot(self.observation_probabilities, a)
            if b is not None:
                b = np.dot(self.observation_probabilities, b)
        # now we are on macrostate space, or something is wrong
        if len(a) == self.nstates:
            return super(HMSM, self).correlation(a, b=b, maxtime=maxtime)
        else:
            raise ValueError(f'observable vectors have size {len(a)} which is incompatible with both hidden ({self.nstates})'
                             f' and observed states ({self.nstates_obs})')

    def fingerprint_correlation(self, a, b=None, k=None, ncv=None):
        # basic checks for a and b
        a = ensure_ndarray(a, ndim=1)
        b = ensure_ndarray(b, ndim=1, size=len(a), allow_None=True)
        # are we on microstates space?
        if len(a) == self.nstates_obs:
            a = np.dot(self.observation_probabilities, a)
            if b is not None:
                b = np.dot(self.observation_probabilities, b)
        # now we are on macrostate space, or something is wrong
        if len(a) == self.nstates:
            return super(HMSM, self).fingerprint_correlation(a, b=b)
        else:
            raise ValueError(f'observable vectors have size {len(a)} which is incompatible with both hidden ({self.nstates})'
                             f' and observed states ({self.nstates_obs})')

    def relaxation(self, p0, a, maxtime=None, k=None, ncv=None):
        # basic checks for a and b
        p0 = ensure_ndarray(p0, ndim=1)
        a = ensure_ndarray(a, ndim=1, size=len(p0))
        # are we on microstates space?
        if len(a) == self.nstates_obs:
            p0 = np.dot(self.observation_probabilities, p0)
            a = np.dot(self.observation_probabilities, a)
        # now we are on macrostate space, or something is wrong
        if len(a) == self.nstates:
            return super(HMSM, self).relaxation(p0, a, maxtime=maxtime)
        else:
            raise ValueError(f'observable vectors have size {len(a)} which is incompatible with both hidden ({self.nstates})'
                             f' and observed states ({self.nstates_obs})')

    def fingerprint_relaxation(self, p0, a, k=None, ncv=None):
        # basic checks for a and b
        p0 = ensure_ndarray(p0, ndim=1)
        a = ensure_ndarray(a, ndim=1, size=len(p0))
        # are we on microstates space?
        if len(a) == self.nstates_obs:
            p0 = np.dot(self.observation_probabilities, p0)
            a = np.dot(self.observation_probabilities, a)
        # now we are on macrostate space, or something is wrong
        if len(a) == self.nstates:
            return super(HMSM, self).fingerprint_relaxation(p0, a)
        else:
            raise ValueError('observable vectors have size %s which is incompatible with both hidden (%s)'
                             ' and observed states (%s)' % (len(a), self.nstates, self.nstates_obs))

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
        M = np.zeros((self.nstates_obs, self.nstates))
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
        for i in range(self.nstates):
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
        #TODO: replace this with numpy.random.choice
        output_distributions = [stats.rv_discrete(values=(np.arange(self.pobs.shape[1]), pobs_i)) for pobs_i in
                                self.pobs]
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
