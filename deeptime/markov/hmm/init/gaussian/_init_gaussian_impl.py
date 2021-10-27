import numpy as np


def from_data(trajs, n_hidden_states, reversible):
    r""" Makes an initial guess :class:`HMM <HiddenMarkovModel>` with Gaussian output model.

    To this end, a Gaussian mixture model is estimated using `scikit-learn <https://scikit-learn.org/>`_.

    Parameters
    ----------
    trajs : array_like or list of array_like
        Trajectories which are used for making the initial guess.
    n_hidden_states : int
        Number of hidden states.
    reversible : bool
        Whether the hidden transition matrix is estimated so that it is reversible.

    Returns
    -------
    hmm_init : HiddenMarkovModel
        An initial guess for the HMM

    See Also
    --------
    deeptime.markov.hmm.GaussianOutputModel : The type of output model this heuristic uses.
    deeptime.markov.hmm.init.discrete.metastable_from_data
    deeptime.markov.hmm.init.discrete.metastable_from_msm
    """
    from deeptime.markov.hmm import HiddenMarkovModel, GaussianOutputModel
    from sklearn.mixture import GaussianMixture
    import deeptime.markov.tools.estimation as msmest
    import deeptime.markov.tools.analysis as msmana
    from deeptime.util.types import ensure_timeseries_data

    trajs = ensure_timeseries_data(trajs)
    collected_observations = np.concatenate(trajs)
    if collected_observations.ndim == 1:
        collected_observations = collected_observations[..., None]
    gmm = GaussianMixture(n_components=n_hidden_states)
    gmm.fit(collected_observations)
    output_model = GaussianOutputModel(n_hidden_states, means=gmm.means_[:, 0], sigmas=np.sqrt(gmm.covariances_[:, 0]))

    # Compute fractional state memberships.
    Nij = np.zeros((n_hidden_states, n_hidden_states))
    for o_t in trajs:
        # length of trajectory
        T = o_t.shape[0]
        # output probability
        pobs = output_model.to_state_probability_trajectory(o_t)
        # normalize
        pobs /= pobs.sum(axis=1)[:, None]
        # Accumulate fractional transition counts from this trajectory.
        for t in range(T - 1):
            Nij += np.outer(pobs[t, :], pobs[t + 1, :])

    # Compute transition matrix maximum likelihood estimate.
    transition_matrix = msmest.transition_matrix(Nij, reversible=reversible)
    initial_distribution = msmana.stationary_distribution(transition_matrix)
    return HiddenMarkovModel(transition_model=transition_matrix, output_model=output_model,
                             initial_distribution=initial_distribution)
