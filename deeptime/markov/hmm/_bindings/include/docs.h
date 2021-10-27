//
// Created by mho on 2/10/20.
//

#pragma once

namespace docs {
static constexpr const char* FORWARD = R"mydelim(Compute P( obs | A, B, pi ) and all forward coefficients.

Parameters
----------
transition_matrix : ndarray((N,N), dtype = float)
    transition matrix of the hidden states
state_probability_trajectory : ndarray((T,N), dtype = float)
    state_probability_trajectory[t,i] is the observation probability for observation at time t given hidden state i
initial_distribution : ndarray((N), dtype = float)
    initial distribution of hidden states
alpha_out : ndarray((T,N), dtype = float)
    container for the alpha result variables. alpha[t,i] is the ith forward coefficient of time t. These can be
    used in many different algorithms related to HMMs.
T : int, optional, default = None
    trajectory length. If not given, T = pobs.shape[0] will be used.

Returns
-------
logprob : float
    The probability to observe the sequence `ob` with the model given by `A`, `B` and `pi`.
)mydelim";

static constexpr const char* BACKWARD = R"mydelim(Compute all backward coefficients. With scaling!
Parameters
----------
transition_matrix : ndarray((N,N), dtype = float)
    transition matrix of the hidden states
state_probability_trajectory : ndarray((T,N), dtype = float)
    pobs[t,i] is the observation probability for observation at time t given hidden state i
beta_out : ndarray((T,N), dtype = float)
    container for the beta result variables. beta[t,i] is the ith backward coefficient of time t. These can be
    used in many different algorithms related to HMMs.
T : int, optional, default = None
    trajectory length. If not given, T = pobs.shape[0] will be used.
)mydelim";

static constexpr const char* STATE_PROBS = R"mydelim(Calculate the (T,N)-probability matrix for being in state i at time t.

Parameters
----------
alpha : ndarray((T,N), dtype = float)
    alpha[t,i] is the ith forward coefficient of time t.
beta : ndarray((T,N), dtype = float)
    beta[t,i] is the ith forward coefficient of time t. gamma[t,i] is the probability at time t to be in state i
gamma_out : ndarray((T,N), dtype = float)
    container for the gamma result variables.
T : int, optional, default = None
    trajectory length. If not given, gamma_out.shape[0] will be used.
See Also
--------
forward : to calculate `alpha`
backward : to calculate `beta`
)mydelim";

static constexpr const char* TRANSITION_COUNTS = R"mydelim(Sum for all t the probability to transition from state i to state j.

Parameters
----------
alpha : ndarray((T,N), dtype = float)
    alpha[t,i] is the ith forward coefficient of time t.
beta : ndarray((T,N), dtype = float)
    beta[t,i] is the ith forward coefficient of time t.
transition_matrix : ndarray((N,N), dtype = float)
    transition matrix of the hidden states
state_probability_trajectory : ndarray((T,N), dtype = float)
    state_probability_trajectory[t,i] is the observation probability for observation at time t given hidden state i
T : int, optional, default=None
    number of time steps, if None set to len(pobs)
counts_out : ndarray((N,N), dtype = float)
    container for the resulting count matrix. counts[i, j] is the summed probability to transition from i to j in time [0,T)

See Also
--------
forward : calculate forward coefficients `alpha`
backward : calculate backward coefficients `beta`
)mydelim";

static constexpr const char* VITERBI = R"mydelim(Estimate the hidden pathway of maximum likelihood using the Viterbi algorithm.

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
)mydelim";

static constexpr const char* SAMPLE_PATH = R"mydelim(Sample the hidden pathway S from the conditional distribution P ( S | Parameters, Observations )

Parameters
----------
alpha : ndarray((T,N), dtype = float)
    alpha[t,i] is the ith forward coefficient of time t.
transition_matrix : ndarray((N,N), dtype = float)
    transition matrix of the hidden states
state_probability_trajectory : ndarray((T,N), dtype = float)
    state_probability_trajectory[t,i] is the observation probability for observation at time t given hidden state i
T : int, optional, default=None
    number of time steps, defaults to len(state_probability_trajectory).
seed: int, optional, default = -1
    seed for libc.stdlib.srand, if -1 is given, the random number generation will be initialized
    with the current time (seconds).

Returns
-------
S : (T,) ndarray
    maximum likelihood hidden path
)mydelim";
}
