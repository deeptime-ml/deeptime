
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
import math

from bhmm.output_models import GaussianOutputModel, DiscreteOutputModel


def generate_transition_matrix(nstates=3, lifetime_max=100, lifetime_min=10, reversible=True):
    """
    Generates random metastable transition matrices

    Parameters
    ----------
    nstates : int, optional, default=3
        Number of states for which row-stockastic transition matrix is to be generated.
    lifetime_max : float, optional, default = 100
        maximum lifetime of any state
    lifetime_min : float, optional, default = 10
        minimum lifetime of any state
    reversible : bool, optional, default=True
        If True, the row-stochastic transition matrix will be reversible.

    Returns
    -------
    Tij : np.array with shape (nstates, nstates)
        A randomly generated row-stochastic transition matrix.

    """
    # regular grid in the log lifetimes
    ltmax = math.log(lifetime_max)
    ltmin = math.log(lifetime_min)
    lt = np.linspace(ltmin, ltmax, num=nstates)
    # create diagonal with self-transition probabilities according to timescales
    diag = 1.0 - 1.0/np.exp(lt)
    # random X
    X = np.random.random((nstates, nstates))
    if reversible:
        X += X.T
    # row-normalize
    T = X / np.sum(X, axis=1)[:, None]
    # enforce lifetimes by rescaling rows
    for i in range(nstates):
        T[i, i] = 0
        T[i, :] *= (1.0-diag[i]) / np.sum(T[i, :])
        T[i, i] = 1.0 - np.sum(T[i, :])

    return T


def force_spectroscopy_model():
    """
    Construct a specific three-state test model intended to be representative of
    single-molecule force spectroscopy experiments.

    Returns
    -------
    model : HMM
        The synthetic HMM model.

    Examples
    --------

    >>> model = force_spectroscopy_model()

    """
    nstates = 3

    # Define state emission probabilities.
    output_model = GaussianOutputModel(nstates, means=[3.0, 4.7, 5.6], sigmas=[1.0, 0.3, 0.2])

    # Define a reversible transition matrix.
    Tij = np.array([[0.98      ,  0.01540412,  0.00459588],
                    [0.06331175,  0.9       ,  0.03668825],
                    [0.00339873,  0.00660127,  0.99      ]])

    # Use stationary distribution as initial distribution
    import msmtools.analysis as msmana
    pi = msmana.stationary_distribution(Tij)

    # Construct HMM with these parameters.
    from bhmm import HMM
    model = HMM(pi, Tij, output_model)

    return model


def dalton_model(nstates=3, omin=-5, omax=5, sigma_min=0.5, sigma_max=2.0,
                 lifetime_max=100, lifetime_min=10, reversible=True, output='gaussian'):
    """
    Construct a test multistate model with regular spaced emission means (linearly interpolated between omin and omax)
    and variable emission widths (linearly interpolated between sigma_min and sigma_max).

    Parameters
    ----------
    nstates : int, optional, default = 3
        number of hidden states
    omin : float, optional, default = -5
        mean position of the first state.
    omax : float, optional, default = 5
        mean position of the last state.
    sigma_min : float, optional, default = 0.5
        The width of the observed gaussian distribution for the first state
    sigma_max : float, optional, default = 2.0
        The width of the observed gaussian distribution for the last state
    lifetime_max : float, optional, default = 100
        maximum lifetime of any state
    lifetime_min : float, optional, default = 10
        minimum lifetime of any state
    reversible : bool, optional, default=True
        If True, the row-stochastic transition matrix will be reversible.
    output_model_type : str, optional, default='gaussian'
        Output model to use, one of ['gaussian', 'discrete']

    Returns
    -------
    model : HMM
        The synthetic HMM model.

    Examples
    --------

    Generate default model.

    >>> model = dalton_model()

    Generate model with specified number of states.

    >>> model = dalton_model(nstates=5)

    Generate non-reversible model.

    >>> model = dalton_model(reversible=False)

    Generate a discrete output model.

    >>> model = dalton_model(output='discrete')

    """

    # parameters
    means = np.linspace(omin, omax, num=nstates)
    sigmas = np.linspace(sigma_min, sigma_max, num=nstates)

    # Define state emission probabilities.
    if output == 'gaussian':
        output_model = GaussianOutputModel(nstates, means=means, sigmas=sigmas)
    elif output == 'discrete':
        # Construct matrix of output probabilities
        # B[i,j] is probability state i produces symbol j, where nsymbols = nstates
        B = np.zeros([nstates, nstates], dtype=np.float64)
        for i in range(nstates):
            for j in range(nstates):
                B[i, j] = np.exp(-0.5 * (means[i] - means[j]) / (sigmas[i] * sigmas[j]))
            B[i, :] /= B[i, :].sum()
        output_model = DiscreteOutputModel(B)
    else:
        raise Exception("output_model_type = '%s' unknown, must be one of ['gaussian', 'discrete']" % output)

    Tij = generate_transition_matrix(nstates, lifetime_max=lifetime_max, lifetime_min=lifetime_min,
                                     reversible=reversible)

    # stationary distribution
    import msmtools.analysis as msmana
    Pi = msmana.stationary_distribution(Tij)

    # Construct HMM with these parameters.
    from bhmm import HMM
    model = HMM(Pi, Tij, output_model)

    return model


def generate_synthetic_observations(nstates=3, ntrajectories=10, length=10000,
                                    omin=-5, omax=5, sigma_min=0.5, sigma_max=2.0,
                                    lifetime_max=100, lifetime_min=10, reversible=True,
                                    output='gaussian'):
    """Generate synthetic data from a random HMM model.

    Parameters
    ----------
    nstates : int, optional, default=3
        The number of states for the underlying HMM model.
    ntrajectories : int, optional, default=10
        The number of synthetic observation trajectories to generate.
    length : int, optional, default=10000
        The length of synthetic observation trajectories to generate.
    omin : float, optional, default = -5
        mean position of the first state.
    omax : float, optional, default = 5
        mean position of the last state.
    sigma_min : float, optional, default = 0.5
        The width of the observed gaussian distribution for the first state
    sigma_max : float, optional, default = 2.0
        The width of the observed gaussian distribution for the last state
    lifetime_max : float, optional, default = 100
        maximum lifetime of any state
    lifetime_min : float, optional, default = 10
        minimum lifetime of any state
    output_model_type : str, optional, default='gaussian'
        Output model to use, one of ['gaussian', 'discrete']

    Returns
    -------
    model : HMM
        The true underlying HMM model.
    O : list of numpy arrays of shape (length)
        The synthetic observation trajectories generated from the HMM model.
    S : list of numpy arrays of shape (length)
        The synthetic state trajectories corresponding to the observation trajectories.

    Examples
    --------

    Generate synthetic observations with default parameters.

    >>> [model, observations, states] = generate_synthetic_observations()

    Generate synthetic observations with discrete state model.

    >>> [model, observations, states] = generate_synthetic_observations(output='discrete')

    """

    # Generate a random HMM model.
    model = dalton_model(nstates, omin=omin, omax=omax, sigma_min=sigma_min, sigma_max=sigma_max,
                         lifetime_max=lifetime_max, lifetime_min=lifetime_min, reversible=reversible,
                         output=output)

    # Generate synthetic data.
    [O, S] = model.generate_synthetic_observation_trajectories(ntrajectories=ntrajectories, length=length)

    return [model, O, S]


def generate_random_bhmm(nstates=3, ntrajectories=10, length=10000,
                         omin=-5, omax=5, sigma_min=0.5, sigma_max=2.0,
                         lifetime_max=100, lifetime_min=10, reversible=True,
                         output='gaussian'):
    """Generate a BHMM model from synthetic data from a random HMM model.

    Parameters
    ----------
    nstates : int, optional, default=3
        The number of states for the underlying HMM model.
    ntrajectories : int, optional, default=10
        The number of synthetic observation trajectories to generate.
    length : int, optional, default=10000
        The length of synthetic observation trajectories to generate.
    omin : float, optional, default = -5
        mean position of the first state.
    omax : float, optional, default = 5
        mean position of the last state.
    sigma_min : float, optional, default = 0.5
        The width of the observed gaussian distribution for the first state
    sigma_max : float, optional, default = 2.0
        The width of the observed gaussian distribution for the last state
    lifetime_max : float, optional, default = 100
        maximum lifetime of any state
    lifetime_min : float, optional, default = 10
        minimum lifetime of any state
    output_model_type : str, optional, default='gaussian'
        Output model to use, one of ['gaussian', 'discrete']

    Returns
    -------
    model : HMM
        The true underlying HMM model.
    O : list of numpy arrays
        The synthetic observation trajectories generated from the HMM model.
    S : list of numpy arrays
        The synthetic state trajectories corresponding to the observation trajectories.
    bhmm : BHMM
        The BHMM model generated.

    Examples
    --------

    Generate BHMM with default parameters.

    >>> model, observations, hidden_traj, bhmm = generate_random_bhmm() # doctest: +SKIP

    Generate BHMM with discerete states.

    >>> model, observations, hidden_traj, bhmm = generate_random_bhmm(output='discrete') # doctest: +SKIP

    """

    # Generate a random HMM model.
    model = dalton_model(nstates, omin=omin, omax=omax, sigma_min=sigma_min, sigma_max=sigma_max,
                         lifetime_max=lifetime_max, lifetime_min=lifetime_min, reversible=reversible,
                         output=output)
    # Generate synthetic data.
    O, S = model.generate_synthetic_observation_trajectories(ntrajectories=ntrajectories, length=length)
    # Initialize a new BHMM model.
    from bhmm import BHMM
    sampled_model = BHMM(O, nstates)

    return model, O, S, sampled_model


def total_state_visits(nstates, S):
    """
    Return summary statistics for state trajectories.

    Parameters
    ----------
    nstates : int
        The number of states.
    S : list of numpy.array
        S[i] is the hidden state trajectory from state i

    """

    N_i = np.zeros(nstates, np.int32)
    min_state = nstates
    max_state = 0
    for s_t in S:
        for state_index in range(nstates):
            N_i[state_index] += (s_t == state_index).sum()
        min_state = min(min_state, s_t.min())
        max_state = max(max_state, s_t.max())
    return N_i, min_state, max_state
