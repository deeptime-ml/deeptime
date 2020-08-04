# This file is part of scikit-time
#
# Copyright (c) 2020 AI4Science Group, Freie Universitaet Berlin (GER)
#
# scikit-time is free software: you can redistribute it and/or modify
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


def from_data(dtrajs, n_hidden_states, reversible):
    r""" Makes an initial guess :class:`HMM <HiddenMarkovModel>` with Gaussian output model.

    To this end, a Gaussian mixture model is estimated using `scikit-learn <https://scikit-learn.org/>`_.

    Parameters
    ----------
    dtrajs : array_like or list of array_like
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
    :class:`GaussianOutputModel <sktime.markov.hmm.GaussianOutputModel>`
        The type of output model this heuristic uses.

    :func:`init.discrete.metastable_from_data <sktime.markov.hmm.init.discrete.metastable_from_data>`
        Initial guess with :class:`Discrete output model <sktime.markov.hmm.DiscreteOutputModel>`.

    :func:`init.discrete.metastable_from_msm <sktime.markov.hmm.init.discrete.metastable_from_msm>`
        Initial guess from an already existing :class:`MSM <sktime.markov.msm.MarkovStateModel>` with discrete
        output model.
    """
    from sktime.markov.hmm import HiddenMarkovModel, GaussianOutputModel
    from sklearn.mixture import GaussianMixture
    from sktime.util import ensure_dtraj_list
    import sktime.markov.tools.estimation as msmest
    import sktime.markov.tools.analysis as msmana

    dtrajs = ensure_dtraj_list(dtrajs)
    collected_observations = np.concatenate(dtrajs)
    gmm = GaussianMixture(n_components=n_hidden_states)
    gmm.fit(collected_observations[:, None])
    output_model = GaussianOutputModel(n_hidden_states, means=gmm.means_[:, 0], sigmas=np.sqrt(gmm.covariances_[:, 0]))

    # Compute fractional state memberships.
    Nij = np.zeros((n_hidden_states, n_hidden_states))
    for o_t in dtrajs:
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
