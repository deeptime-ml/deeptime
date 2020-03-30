import numpy as np


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
