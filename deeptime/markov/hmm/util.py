import numpy as np
from . import _hmm_bindings as _bindings


def observations_in_state(hidden_state_trajectories, observed_state_trajectories, hidden_state):
    dtype = observed_state_trajectories[0].dtype
    collected_observations = np.array([], dtype=dtype)
    for (s_t, o_t) in zip(hidden_state_trajectories, observed_state_trajectories):
        indices = np.where(s_t == hidden_state)[0]
        collected_observations = np.append(collected_observations, o_t[indices])

    # collected_observations = [
    #    o_t[np.where(s_t == state_index)[0]] for s_t, o_t in zip(self.hidden_state_trajectories, observations)
    # ]
    # return np.hstack(collected_observations)
    return collected_observations


def sample_hidden_state_trajectory(transition_matrix, output_model, initial_distribution, obs, temp_alpha=None):
    """Sample a hidden state trajectory from the conditional distribution P(s | T, E, o)

    Parameters
    ----------
    transition_matrix : (n, n) ndarray
        The transition matrix :math:`T` over the hidden states.
    output_model : deeptime.markov.hmm.OutputModel
        Output model with emission probabilities :math:`E`.
    initial_distribution : (n,) ndarray
        Initial distribution over hidden states.
    obs : (T,) ndarray
        Trajectory in observation space.
    temp_alpha : (T, n) ndarray, optional, default=None
        Optional array that is used to store alphas from the forward pass, if provided it is used for storage
        instead of allocating new memory.

    Returns
    -------
    s_t : (T,) ndarray
        Hidden state trajectory, with s_t[t] the hidden state corresponding to observation obs[t]
    """

    # Determine observation trajectory length
    T = obs.shape[0]

    if temp_alpha is None:
        temp_alpha = np.zeros((obs.shape[0], transition_matrix.shape[0]), dtype=transition_matrix.dtype)

    # compute output probability matrix
    pobs = output_model.to_state_probability_trajectory(obs)
    # compute forward variables
    _bindings.util.forward(transition_matrix, pobs, initial_distribution, T=T, alpha_out=temp_alpha)
    # sample path
    S = _bindings.util.sample_path(temp_alpha, transition_matrix, T=T)
    return S
