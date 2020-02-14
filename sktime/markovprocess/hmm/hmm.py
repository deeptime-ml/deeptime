from typing import Union, Optional, List

import numpy as np
import sktime.markovprocess.hmm._hmm_bindings as _bindings

from sktime.base import Model
from sktime.markovprocess import MarkovStateModel
from sktime.markovprocess.hmm.output_model import OutputModel, DiscreteOutputModel
from sktime.util import ensure_dtraj_list


class HiddenMarkovStateModel(Model):

    def __init__(self, transition_model: Union[np.ndarray, MarkovStateModel],
                 output_model: Union[np.ndarray, OutputModel],
                 initial_distribution: Optional[np.ndarray] = None,
                 likelihoods: Optional[np.ndarray] = None,
                 state_probabilities: Optional[List[np.ndarray]] = None,
                 initial_count : Optional[np.ndarray] = None,
                 hidden_state_trajectories : Optional[List[np.ndarray]] = None,
                 stride: Union[int, str] = 1,
                 observation_symbols: Optional[np.ndarray] = None):
        r"""
        Constructs a new hidden markov state model from a (m, m) hidden transition matrix (macro states), an
        observation probability matrix that maps from hidden to observable states (micro states), i.e., a (m, n)-matrix,
        and an initial distribution over the hidden states.

        Parameters
        ----------
        transition_model : (m,m) ndarray or MarkovStateModel
            Transition matrix for hidden (macro) states
        output_model : (m,n) ndarray or OutputModel
            observation probability matrix from hidden to observable (micro) states or OutputModel instance which yields
            the mapping from hidden to observable state.
        initial_distribution : (m,) ndarray, optional, default=None
            Initial distribution of the hidden (macro) states. Default is uniform.
        likelihoods : (k,) ndarray, optional, default=None
            Likelihood progression of the HMM as it was trained for k iterations with Baum-Welch.
        state_probabilities : list of ndarray, optional, default=None
            List of state probabilities for each trajectory that the model was trained on (gammas).
        initial_count : ndarray, optional, default=None
            Initial counts of the hidden (macro) states, computed from the gamma output of the Baum-Welch algorithm
        """
        if isinstance(transition_model, np.ndarray):
            transition_model = MarkovStateModel(transition_model)
        if isinstance(output_model, np.ndarray):
            output_model = DiscreteOutputModel(output_model)
        if transition_model.n_states != output_model.n_hidden_states:
            raise ValueError("Transition model must describe hidden states")
        if initial_distribution is None:
            # uniform
            initial_distribution = np.ones(transition_model.n_states) / transition_model.n_states
        if initial_distribution.shape[0] != transition_model.n_states:
            raise ValueError("Initial distribution over hidden states must be of length {}"
                             .format(transition_model.n_states))
        self._transition_model = transition_model
        self._output_model = output_model
        self._initial_distribution = initial_distribution
        self._likelihoods = likelihoods
        self._state_probabilities = state_probabilities
        self._initial_count = initial_count
        self._hidden_state_trajectories = hidden_state_trajectories
        if observation_symbols is None and output_model.n_observable_states >= 0:
            observation_symbols = np.arange(output_model.n_observable_states)
        self._observation_symbols = observation_symbols
        self._stride = stride

    @property
    def stride(self):
        return self._stride

    @property
    def observation_symbols(self) -> Optional[np.ndarray]:
        r"""
        Property to get the symbols represented by this HMM in observation space. Can be None in case the output
        model has no discrete observations it assigns to.

        Returns
        -------
        The list of observation symbols or None.
        """
        return self._observation_symbols

    @property
    def transition_model(self) -> MarkovStateModel:
        return self._transition_model

    @property
    def output_model(self) -> OutputModel:
        return self._output_model

    @property
    def initial_distribution(self) -> np.ndarray:
        return self._initial_distribution

    @property
    def n_hidden_states(self):
        return self.output_model.n_hidden_states

    @property
    def likelihoods(self) -> Optional[np.ndarray]:
        return self._likelihoods

    @property
    def likelihood(self) -> Optional[float]:
        if self.likelihoods is not None:
            return self.likelihoods[-1]
        return None

    @property
    def state_probabilities(self) -> Optional[List[np.ndarray]]:
        return self._state_probabilities

    @property
    def transition_counts(self) -> Optional[np.ndarray]:
        return self.transition_model.count_model.count_matrix if self.transition_model.count_model is not None else None

    @property
    def initial_count(self) -> Optional[np.ndarray]:
        return self._initial_count

    @property
    def hidden_state_trajectories(self) -> Optional[List[np.ndarray]]:
        return self._hidden_state_trajectories

    @property
    def output_probabilities(self):
        if isinstance(self.output_model, DiscreteOutputModel):
            return self.output_model.output_probabilities
        raise ValueError("Output probabilities are only available for HMMs with discrete output model.")

    @property
    def n_states_obs(self):
        return self.output_probabilities.shape[1]

    @property
    def stationary_distribution_obs(self):
        if isinstance(self.output_model, DiscreteOutputModel):
            return np.dot(self.transition_model.stationary_distribution, self.output_probabilities)
        raise RuntimeError("only available for discrete output model")

    @property
    def eigenvectors_left_obs(self):
        return np.dot(self.transition_model.eigenvectors_left(), self.output_probabilities)

    @property
    def eigenvectors_right_obs(self):
        return np.dot(self.metastable_memberships, self.transition_model.eigenvectors_right())

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
        Pi_c = np.diag(self.transition_model.stationary_distribution)
        P_c = self.transition_model.transition_matrix
        P_c_k = np.linalg.matrix_power(P_c, k)  # take a power if needed
        B = self.output_probabilities
        C = np.dot(np.dot(B.T, Pi_c), np.dot(P_c_k, B))
        P = C / C.sum(axis=1)[:, None]  # row normalization
        return P

    @property
    def lifetimes(self):
        return -self.transition_model.lagtime / np.log(np.diag(self.transition_model.transition_matrix))

    def compute_viterbi_paths(self, observations: List[np.ndarray]):
        """Computes the Viterbi paths using the current HMM model"""
        observations = ensure_dtraj_list(observations)
        A = self.transition_model.transition_matrix
        pi = self.initial_distribution
        state_probabilities = self.output_model.to_state_probability_trajectory(observations)
        paths = [viterbi(A, obs, pi) for obs in state_probabilities]
        return paths

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
            states = np.arange(self.n_hidden_states)
        if obs is None:
            obs = np.arange(self.n_states_obs)

        transition_model = self.transition_model.submodel(states)

        initial_count = self.initial_count[states].copy()
        initial_distribution = self.initial_distribution[states] / np.sum(self.initial_distribution[states])

        # observation matrix
        output_model = self.output_model.submodel(states, obs)

        observation_symbols = self.observation_symbols
        if observation_symbols is not None:
            observation_symbols = observation_symbols[obs]

        model = HiddenMarkovStateModel(transition_model=transition_model, output_model=output_model,
                                       initial_distribution=initial_distribution, likelihoods=self.likelihoods,
                                       state_probabilities=self.state_probabilities, initial_count=initial_count,
                                       hidden_state_trajectories=self.hidden_state_trajectories,
                                       observation_symbols=observation_symbols)
        return model

    def _select_states(self, connectivity_threshold, states):
        if str(connectivity_threshold) == '1/n':
            connectivity_threshold = 1.0 / float(self.transition_model.n_states)
        if isinstance(states, str):
            strong = 'strong' in states
            largest = 'largest' in states
            count_model = self.transition_model.count_model
            S = count_model.connected_sets(connectivity_threshold=connectivity_threshold, directed=strong)
            if largest:
                score = np.array([len(s) for s in S])
            else:
                score = np.array([count_model.count_matrix[np.ix_(s, s)].sum() for s in S])
            states = S[np.argmax(score)]
        return states

    def nonempty_obs(self, dtrajs):
        from sktime.markovprocess.util import compute_dtrajs_effective, count_states
        if dtrajs is None:
            raise ValueError("Needs nonempty dtrajs to evaluate nonempty obs.")
        dtrajs = ensure_dtraj_list(dtrajs)
        dtrajs_lagged_strided = compute_dtrajs_effective(
            dtrajs, self.transition_model.lagtime, self.transition_model.count_model.n_states_full, self.stride
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
        lcc = self.transition_model.count_model.connected_sets(connectivity_threshold=connectivity_threshold)[0]
        return self.submodel(lcc)

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
        M = np.zeros((self.n_states_obs, self.transition_model.n_states))
        M[nonzero, :] = np.transpose(np.diag(self.transition_model.stationary_distribution).dot(
            self.output_probabilities[:, nonzero]) / self.stationary_distribution_obs[nonzero])
        # renormalize
        M[nonzero, :] /= M.sum(axis=1)[nonzero, None]
        return M

    @property
    def metastable_distributions(self):
        r""" Returns the output probability distributions. Identical to
            :meth:`output_probabilities`

        Returns
        -------
        Pout : ndarray (m,n)
            output probability matrix from hidden to observable discrete states

        See also
        --------
        output_probabilities

        """
        return self.output_probabilities

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
        for i in range(self.transition_model.n_states):
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

        See Also
        --------
        output_probabilities

        """
        return np.argmax(self.output_probabilities, axis=0)


def viterbi(transition_matrix: np.ndarray, state_probability_trajectory: np.ndarray, initial_distribution: np.ndarray):
    """ Estimate the hidden pathway of maximum likelihood using the Viterbi algorithm.

    Parameters
    ----------
    transition_matrix : ndarray((N,N), dtype = float)
        transition matrix of the hidden states
    state_probability_trajectory : ndarray((T,N), dtype = float)
        pobs[t,i] is the observation probability for observation at time t given hidden state i
    initial_distribution : ndarray((N), dtype = float)
        initial distribution of hidden states

    Returns
    -------
    q : numpy.array shape (T)
        maximum likelihood hidden path

    """
    return _bindings.util.viterbi(transition_matrix=transition_matrix,
                                  state_probability_trajectory=state_probability_trajectory,
                                  initial_distribution=initial_distribution)
