# This file is part of BHMM (Bayesian Hidden Markov Models).
#
# Copyright (c) 2016 Frank Noe (Freie Universitaet Berlin)
# and John D. Chodera (Memorial Sloan-Kettering Cancer Center, New York)
#
# BHMM is free software: you can redistribute it and/or modify
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

from sktime.markovprocess.bhmm import HMM
from sktime.markovprocess.bhmm.output_models.gaussian import GaussianOutputModel


def init_model_gaussian1d(observations, n_states, lag, reversible=True):
    """Generate an initial model with 1D-Gaussian output densities

    Parameters
    ----------
    observations : list of ndarray((T_i), dtype=float)
        list of arrays of length T_i with observation data
    n_states : int
        The number of states.

    Examples
    --------

    Generate initial model for a gaussian output model.

    >>> from sktime.markovprocess.bhmm import testsystems
    >>> model, observations, states = testsystems.generate_synthetic_observations(output='gaussian')
    >>> initial_model = init_model_gaussian1d(observations, model.n_states, lag=1)

    """
    # Concatenate all observations.
    collected_observations = np.concatenate(observations)

    # Fit a Gaussian mixture model to obtain emission distributions and state stationary probabilities.
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=n_states)
    gmm.fit(collected_observations[:, None])
    output_model = GaussianOutputModel(n_states, means=gmm.means_[:, 0], sigmas=np.sqrt(gmm.covariances_[:, 0]))

    # Compute fractional state memberships.
    Nij = np.zeros((n_states, n_states))
    for o_t in observations:
        # length of trajectory
        T = o_t.shape[0]
        # output probability
        pobs = output_model.p_obs(o_t)
        # normalize
        pobs /= pobs.sum(axis=1)[:, None]
        # Accumulate fractional transition counts from this trajectory.
        for t in range(T - 1):
            Nij += np.outer(pobs[t, :], pobs[t + 1, :])

    # Compute transition matrix maximum likelihood estimate.
    import msmtools.estimation as msmest
    import msmtools.analysis as msmana
    Tij = msmest.transition_matrix(Nij, reversible=reversible)
    pi = msmana.stationary_distribution(Tij)

    # Update model.
    model = HMM(pi, Tij, output_model, lag=lag)

    return model
