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
from numbers import Integral
from typing import Optional, Union

import numpy as np

from sktime.markovprocess import MarkovStateModel, TransitionCountModel
from sktime.markovprocess.bhmm.hmm.generic_hmm import HMM as BHMM_HMM
from sktime.markovprocess.util import count_states, compute_dtrajs_effective
from sktime.numeric import mdot
from sktime.util import ensure_ndarray, ensure_dtraj_list


class HiddenMarkovStateModel(MarkovStateModel):
    r""" Hidden Markov model on discrete states.
    """

    def __init__(self, transition_matrix, observation_probabilities, stride=1, stationary_distribution=None,
                 n_eigenvalues=None, reversible=None, count_model=None, initial_distribution=None, initial_counts=None,
                 ncv: Optional[int] = None, bhmm_model: BHMM_HMM = None, observation_state_symbols=None,
                 n_observation_states_full=None):
        r"""
        Constructs a new hidden markov state model from a hidden transition matrix (micro states) and an observation
        probability matrix that maps from hidden to observable discrete states (macro states).

        Parameters
        ----------
        transition_matrix : ndarray (m,m)
            macro-state or hidden transition matrix
        observation_probabilities : ndarray (m,n)
            observation probability matrix from hidden to observable discrete states (micro states)
        stride : int or str('effective'), optional, default=1
            Stride which was used to subsample discrete trajectories while estimating a HMSM. Can either be an integer
            value which determines the offset or 'effective', which makes an estimate of a stride at which subsequent
            discrete trajectory elements are uncorrelated.
        stationary_distribution : ndarray(m), optional, default=None
            Stationary distribution. Can be optionally given in case if it was
            already computed, e.g. by the estimator.
        n_eigenvalues : int or None
            The number of eigenvalues / eigenvectors to be kept. If set to None,
            defaults will be used. For a dense MarkovStateModel the default is all eigenvalues.
            For a sparse MarkovStateModel the default is 10.
        reversible : bool, optional, default=None
            whether P is reversible with respect to its stationary distribution.
            If None (default), will be determined from P
        count_model : TransitionCountModel, optional, default=None
            Transition count model containing count matrix and potentially data statistics for the hidden (macro)
            states. Not required for instantiation, default is None.
        initial_distribution : ndarray(m), optional, default=None
            Initial distribution of the hidden (macro) states
        initial_counts : ndarray(m), optional, default=None
            Initial counts of the hidden (macro) states, computed from the gamma output of the Baum-Welch algorithm
        ncv : int, optional, default=None
            Relevant for eigenvalue decomposition of reversible transition
            matrices. It is the number of Lanczos vectors generated, `ncv` must
            be greater than n_eigenvalues; it is recommended that ncv > 2*neig.
        bhmm_model : BHMM_HMM, optional, default=None
            bhmm hmm model TODO to be removed
        observation_state_symbols : array_like of int, optional, default=None
            Sorted unique symbols in observations. If None, it is assumed that all possible observations are made
            and the state symbols are set to an iota range over the number of observation states.
        n_observation_states_full : int, optional, default=None
            Number of possible observation states. It is assumed that the symbols form a iota range from 0 (inclusive)
            to n_observation_states_full (exclusive). If None, it is assumed that the full set of observation states
            is captured by this model and is set to n_observation_states.
        """
        super(HiddenMarkovStateModel, self).__init__(
            transition_matrix=transition_matrix, stationary_distribution=stationary_distribution,
            reversible=reversible, n_eigenvalues=n_eigenvalues, ncv=ncv, count_model=count_model
        )

        observation_probabilities = ensure_ndarray(observation_probabilities, ndim=2, dtype=np.float64)
        assert np.allclose(observation_probabilities.sum(axis=1), 1), 'pobs is not a stochastic matrix'
        self._n_states_obs = observation_probabilities.shape[1]
        self._observation_probabilities = observation_probabilities
        self._initial_distribution = initial_distribution
        self._initial_counts = initial_counts
        self._hmm = bhmm_model
        if observation_state_symbols is None:
            # iota range over n observation states, already sorted
            self._observation_state_symbols = np.arange(self.n_observation_states)
        else:
            # sort observation states and set member
            self._observation_state_symbols = np.sort(observation_state_symbols)
        if n_observation_states_full is None:
            n_observation_states_full = self.n_observation_states
        self._n_observation_states_full = n_observation_states_full
        if not (isinstance(stride, Integral) or (isinstance(stride, str) and stride == 'effective')):
            raise ValueError("Stride argument must either be an integer value or 'effective', "
                             "but was: {}".format(stride))
        self._stride = stride

    @property
    def stride(self) -> Union[Integral, str]:
        r"""
        The stride parameter which was used to subsample the discrete trajectories when estimating the hidden
        markov state model. Can either be an integer value or 'effective', in which case a stride is estimated at
        which subsequent states are uncorrelated.

        Returns
        -------
        The stride parameter.
        """
        return self._stride

    @property
    def n_observation_states_full(self):
        r"""
        Yields the total number of observation states ignoring whether this HMSM represents only a subset of them.
        It is assumed that the possible observation states form a iota range from 0 (inclusive) to
        n_observation_states_full (exclusive).

        Returns
        -------
        The full number of observation states.
        """
        return self._n_observation_states_full

    @property
    def observation_state_symbols(self):
        r"""
        Observation symbols that are represented in this hidden markov model. This can be a subset of all possible
        observations in the trajectories.

        Returns
        -------
        List of observation symbols represented in this model, sorted.
        """
        return self._observation_state_symbols

    @property
    def n_observation_states(self):
        r"""
        Property determining the number of observed/macro states. It coincides with the size of the second axis
        of the observation probabilities matrix.

        Returns
        -------
        Number of observed/macro states
        """
        return self.observation_probabilities.shape[1]

    @property
    def count_model(self) -> Optional[TransitionCountModel]:
        r"""
        Yields the count model for the micro (hidden) states. The count matrix is estimated from Viterbi paths.

        Returns
        -------
        The count model for the micro states.
        """
        return super().count_model

    @property
    def initial_counts(self) -> np.ndarray:
        """
        Hidden initial counts.
        Returns
        -------
        initial counts
        """
        return self._initial_counts

    @initial_counts.setter
    def initial_counts(self, value):
        self._initial_counts = value

    @property
    def bhmm_model(self) -> BHMM_HMM:
        return self._hmm

    ################################################################################
    # Submodel functions using estimation information (counts)
    ################################################################################

    def submodel(self, states: Optional[np.ndarray] = None, obs: Optional[np.ndarray] = None):
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
        Returns
        -------
        hmm : HiddenMarkovStateModel
            The restricted HMM.
        """

        if states is None and obs is None:
            return self  # do nothing
        if states is None:
            states = np.arange(self.n_states)
        if obs is None:
            obs = np.arange(self.n_states_obs)

        count_model = self.count_model
        if count_model is not None:
            from sktime.markovprocess.bhmm.estimators import _tmatrix_disconnected
            count_model = count_model.submodel(states)
            P = _tmatrix_disconnected.estimate_P(count_model.count_matrix, reversible=self.reversible,
                                                 mincount_connectivity=0)
            P /= P.sum(axis=1)[:, None]
            stationary_distribution = _tmatrix_disconnected.stationary_distribution(P, count_model.count_matrix)
        else:
            P = self.transition_matrix[np.ix_(states, states)].copy()
            P /= P.sum(axis=1)[:, None]

            stationary_distribution = self.stationary_distribution
            if stationary_distribution is not None:
                stationary_distribution = stationary_distribution[states]

        initial_count = self.initial_counts[states].copy()
        initial_distribution = self.initial_distribution[states] / self.initial_distribution[states].sum()

        # observation matrix
        B = self.observation_probabilities[np.ix_(states, obs)].copy()
        B /= B.sum(axis=1)[:, None]

        symbols = self.observation_state_symbols[obs]

        model = HiddenMarkovStateModel(
            transition_matrix=P, observation_probabilities=B, stride=self.stride,
            stationary_distribution=stationary_distribution, n_eigenvalues=self.n_eigenvalues,
            reversible=self.reversible, count_model=count_model, initial_counts=initial_count,
            initial_distribution=initial_distribution, ncv=self.ncv, bhmm_model=self.bhmm_model,
            observation_state_symbols=symbols, n_observation_states_full=self.n_observation_states_full)
        return model

    def _select_states(self, connectivity_threshold, states):
        if str(connectivity_threshold) == '1/n':
            connectivity_threshold = 1.0 / float(self.n_states)
        if isinstance(states, str):
            strong = 'strong' in states
            largest = 'largest' in states
            S = self.count_model.connected_sets(connectivity_threshold=connectivity_threshold, directed=strong)
            if largest:
                score = np.array([len(s) for s in S])
            else:
                score = np.array([self.count_model.count_matrix[np.ix_(s, s)].sum() for s in S])
            states = S[np.argmax(score)]
        return states

    def nonempty_obs(self, dtrajs):
        if dtrajs is None:
            raise ValueError("Needs nonempty dtrajs to evaluate nonempty obs.")
        dtrajs = ensure_dtraj_list(dtrajs)
        dtrajs_lagged_strided = compute_dtrajs_effective(
            dtrajs, self.count_model.lagtime, self.count_model.n_states_full, self.stride
        )
        obs = np.where(count_states(dtrajs_lagged_strided) > 0)[0]
        return obs

    def states_largest(self, strong=True, connectivity_threshold='1/n'):
        return self._select_states(connectivity_threshold, 'largest-strong' if strong else 'largest-weak')

    def submodel_largest(self, strong=True, connectivity_threshold='1/n', observe_nonempty=True, dtrajs=None):
        """ Returns the largest connected sub-HMM (convenience function)

        Returns
        -------
        hmm : HiddenMarkovStateModel
            The restricted HMSM.

        """
        states = self.states_largest(strong=strong, connectivity_threshold=connectivity_threshold)
        obs = self.nonempty_obs(dtrajs) if observe_nonempty else None
        return self.submodel(states=states, obs=obs)

    def states_populous(self, strong=True, connectivity_threshold='1/n'):
        return self._select_states(connectivity_threshold, 'populous-strong' if strong else 'populous-weak')

    def submodel_populous(self, strong=True, connectivity_threshold='1/n', observe_nonempty=True, dtrajs=None):
        """ Returns the most populous connected sub-HMM (convenience function)

        Returns
        -------
        hmm : HiddenMarkovStateModel
            The restricted HMSM.

        """
        states = self.states_populous(strong=strong, connectivity_threshold=connectivity_threshold)
        obs = self.nonempty_obs(dtrajs) if observe_nonempty else None
        return self.submodel(states=states, obs=obs)

    def submodel_disconnect(self, connectivity_threshold='1/n'):
        """Disconnects sets of hidden states that are barely connected

        Runs a connectivity check excluding all transition counts below
        connectivity_threshold. The transition matrix and stationary distribution
        will be re-estimated. Note that the resulting transition matrix
        may have both strongly and weakly connected subsets.

        Parameters
        ----------
        connectivity_threshold : float or '1/n'
            minimum number of counts to consider a connection between two states.
            Counts lower than that will count zero in the connectivity check and
            may thus separate the resulting transition matrix. The default
            evaluates to 1/n_states.

        Returns
        -------
        hmm : HiddenMarkovStateModel
            The restricted HMM.

        """
        lcc = self.count_model.connected_sets(connectivity_threshold=connectivity_threshold)[0]
        return self.submodel(lcc)

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
        return -self.count_model.physical_time / np.log(np.diag(self.transition_matrix))

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
            return super(HiddenMarkovStateModel, self).expectation(a)
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
            return super(HiddenMarkovStateModel, self).correlation(a, b=b, maxtime=maxtime)
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
            return super(HiddenMarkovStateModel, self).fingerprint_correlation(a, b=b)
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
            return super(HiddenMarkovStateModel, self).relaxation(p0, a, maxtime=maxtime)
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
            return super(HiddenMarkovStateModel, self).fingerprint_relaxation(p0, a)
        else:
            raise ValueError('observable vectors have size %s which is incompatible with both hidden (%s)'
                             ' and observed states (%s)' % (len(a), self.n_states, self.n_states_obs))

    def pcca(self, n_metastable_sets):
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
        output_distributions = [stats.rv_discrete(values=(np.arange(self._observation_probabilities.shape[1]), pobs_i))
                                for pobs_i in
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

    # TODO: generate_traj. How should that be defined? Probably indexes of observable states, but should we specify
    #                      hidden or observable states as start and stop states?
    # TODO: sample_by_state. How should that be defined?

    def transform_discrete_trajectories_to_observed_symbols(self, dtrajs):
        r"""A list of integer arrays with the discrete trajectories mapped to the currently used set of observation
        symbols. For example, if there has been a subselection of the model for connectivity='largest', the indices
        will be given within the connected set, frames that do not correspond to a considered symbol are set to -1.

        Parameters
        ----------
        dtrajs : array_like or list of array_like
            discretized trajectories

        Returns
        -------
        array_like or list of array_like
            Curated discretized trajectories so that unconsidered symbols are mapped to -1.
        """

        dtrajs = ensure_dtraj_list(dtrajs)
        mapping = -1 * np.ones(self.n_observation_states_full, dtype=np.int32)
        mapping[self.observation_state_symbols] = np.arange(self.n_observation_states)
        return [mapping[dtraj] for dtraj in dtrajs]

    def sample_by_observation_probabilities(self, dtrajs, nsample):
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
        from sktime.markovprocess.sample import compute_index_states
        mapped = self.transform_discrete_trajectories_to_observed_symbols(dtrajs)
        observable_state_indices = compute_index_states(mapped)
        return sample_indexes_by_distribution(observable_state_indices, self.observation_probabilities, nsample)
