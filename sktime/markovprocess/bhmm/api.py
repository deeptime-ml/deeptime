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

import numpy as _np
from msmtools.util import types


def _guess_output_type(observations):
    """ Suggests a HMM model type based on the observation data

    Uses simple rules in order to decide which HMM model type makes sense based on observation data.
    If observations consist of arrays/lists of integer numbers (irrespective of whether the python type is
    int or float), our guess is 'discrete'.
    If observations consist of arrays/lists of 1D-floats, our guess is 'discrete'.
    In any other case, a TypeError is raised because we are not supporting that data type yet.

    Parameters
    ----------
    observations : list of lists or arrays
        observation trajectories

    Returns
    -------
    output_type : str
        One of {'discrete', 'gaussian'}

    """
    o1 = _np.array(observations[0])

    # CASE: vector of int? Then we want a discrete HMM
    if types.is_int_vector(o1):
        return 'discrete'

    # CASE: not int type, but everything is an integral number. Then we also go for discrete
    if _np.allclose(o1, _np.round(o1)):
        isintegral = True
        for i in range(1, len(observations)):
            if not _np.allclose(observations[i], _np.round(observations[i])):
                isintegral = False
                break
        if isintegral:
            return 'discrete'

    # CASE: vector of double? Then we want a gaussian
    if types.is_float_vector(o1):
        return 'gaussian'

    # None of the above? Then we currently do not support this format!
    raise TypeError('Observations is neither sequences of integers nor 1D-sequences of floats. The current version'
                    'does not support your input.')


def gaussian_hmm(pi, P, means, sigmas):
    """ Initializes a 1D-Gaussian HMM

    Parameters
    ----------
    pi : ndarray(n_states, )
        Initial distribution.
    P : ndarray(n_states,n_states)
        Hidden transition matrix
    means : ndarray(n_states, )
        Means of Gaussian output distributions
    sigmas : ndarray(n_states, )
        Standard deviations of Gaussian output distributions
    stationary : bool, optional, default=True
        If True: initial distribution is equal to stationary distribution of transition matrix
    reversible : bool, optional, default=True
        If True: transition matrix will fulfill detailed balance constraints.

    """
    from .output_models.gaussian import GaussianOutputModel
    from .hmm.generic_hmm import HMM
    # count states
    n_states = _np.array(P).shape[0]
    # initialize output model
    output_model = GaussianOutputModel(n_states, means, sigmas)
    # initialize general HMM
    ghmm = HMM(pi, P, output_model)
    return ghmm


def discrete_hmm(pi, P, pout):
    """ Initializes a discrete HMM

    Parameters
    ----------
    pi : ndarray(n_states, )
        Initial distribution.
    P : ndarray(n_states,n_states)
        Hidden transition matrix
    pout : ndarray(n_states,nsymbols)
        Output matrix from hidden states to observable symbols
    """
    from .output_models.discrete import DiscreteOutputModel
    from .hmm.generic_hmm import HMM

    # initialize output model
    output_model = DiscreteOutputModel(pout)
    # initialize general HMM
    dhmm = HMM(pi, P, output_model)
    return dhmm


def init_hmm(observations, n_states, lag=1, output=None, reversible=True):
    """Use a heuristic scheme to generate an initial model.

    Parameters
    ----------
    observations : list of ndarray((T_i))
        list of arrays of length T_i with observation data
    n_states : int
        The number of states.
    output : str, optional, default=None
        Output model type from [None, 'gaussian', 'discrete']. If None, will automatically select an output
        model type based on the format of observations.

    Examples
    --------

    Generate initial model for a gaussian output model.

    >>> from sktime.markovprocess import bhmm
    >>> model, observations, states = bhmm.testsystems.generate_synthetic_observations(output='gaussian')
    >>> initial_model = init_hmm(observations, model.n_states, output='gaussian')

    Generate initial model for a discrete output model.

    >>> from sktime.markovprocess import bhmm
    >>> model, observations, states = bhmm.testsystems.generate_synthetic_observations(output='discrete')
    >>> initial_model = init_hmm(observations, model.n_states, output='discrete')

    """
    # select output model type
    if output is None:
        output = _guess_output_type(observations)

    if output == 'discrete':
        return init_discrete_hmm(observations, n_states, lag=lag, reversible=reversible)
    elif output == 'gaussian':
        return init_gaussian_hmm(observations, n_states, lag=lag, reversible=reversible)
    else:
        raise NotImplementedError('output model type ' + str(output) + ' not yet implemented.')


def init_gaussian_hmm(observations, n_states, lag=1, reversible=True):
    """ Use a heuristic scheme to generate an initial model.

    Parameters
    ----------
    observations : list of ndarray((T_i))
        list of arrays of length T_i with observation data
    n_states : int
        The number of states.

    Examples
    --------

    Generate initial model for a gaussian output model.

    >>> from sktime.markovprocess import bhmm
    >>> [model, observations, states] = bhmm.testsystems.generate_synthetic_observations(output='gaussian')
    >>> initial_model = init_gaussian_hmm(observations, model.n_states)

    """
    from .init import gaussian
    if lag > 1:
        observations = lag_observations(observations, lag)
    hmm0 = gaussian.init_model_gaussian1d(observations, n_states, lag, reversible=reversible)
    return hmm0


# TODO: remove lag here?
def init_discrete_hmm(observations, n_states, lag=1, reversible=True, stationary=True, regularize=True,
                      method='connect-spectral', separate=None):
    """Use a heuristic scheme to generate an initial model.

    Parameters
    ----------
    observations : list of ndarray((T_i))
        list of arrays of length T_i with observation data
    n_states : int
        The number of states.
    lag : int
        Lag time at which the observations should be counted.
    reversible : bool
        Estimate reversible HMM transition matrix.
    stationary : bool
        p0 is the stationary distribution of P. Currently only reversible=True is implemented
    regularize : bool
        Regularize HMM probabilities to avoid 0's.
    method : str
        * 'lcs-spectral' : Does spectral clustering on the largest connected set
            of observed states.
        * 'connect-spectral' : Uses a weak regularization to connect the weakly
            connected sets and then initializes HMM using spectral clustering on
            the nonempty set.
        * 'spectral' : Uses spectral clustering on the nonempty subsets. Separated
            observed states will end up in separate hidden states. This option is
            only recommended for small observation spaces. Use connect-spectral for
            large observation spaces.
    separate : None or iterable of int
        Force the given set of observed states to stay in a separate hidden state.
        The remaining n_states-1 states will be assigned by a metastable decomposition.

    Examples
    --------

    Generate initial model for a discrete output model.

    >>> from sktime.markovprocess import bhmm
    >>> model, observations, states = bhmm.testsystems.generate_synthetic_observations(output='discrete')
    >>> initial_model = init_discrete_hmm(observations, model.n_states)

    """
    import msmtools.estimation as msmest
    from .init.discrete import init_discrete_hmm_spectral
    C = msmest.count_matrix(observations, lag, sparse_return=False)
    # regularization
    if regularize:
        eps_A = None
        eps_B = None
    else:
        eps_A = 0
        eps_B = 0
    if not stationary:
        raise NotImplementedError('Discrete-HMM initialization with stationary=False is not yet implemented.')

    if method == 'lcs-spectral':
        lcs = msmest.largest_connected_set(C)
        p0, P, B = init_discrete_hmm_spectral(C, n_states, reversible=reversible, stationary=stationary,
                                              active_set=lcs, separate=separate, eps_A=eps_A, eps_B=eps_B)
    elif method == 'connect-spectral':
        # make sure we're strongly connected
        C += msmest.prior_neighbor(C, 0.001)
        nonempty = _np.where(C.sum(axis=0) + C.sum(axis=1) > 0)[0]
        C[nonempty, nonempty] = _np.maximum(C[nonempty, nonempty], 0.001)
        p0, P, B = init_discrete_hmm_spectral(C, n_states, reversible=reversible, stationary=stationary,
                                              active_set=nonempty, separate=separate, eps_A=eps_A, eps_B=eps_B)
    elif method == 'spectral':
        p0, P, B = init_discrete_hmm_spectral(C, n_states, reversible=reversible, stationary=stationary,
                                              active_set=None, separate=separate, eps_A=eps_A, eps_B=eps_B)
    else:
        raise NotImplementedError('Unknown discrete-HMM initialization method ' + str(method))

    hmm0 = discrete_hmm(p0, P, B)
    hmm0._lag = lag
    return hmm0


# TODO: remove lag here?
def estimate_hmm(observations, n_states, lag=1, initial_model=None, output=None,
                 reversible=True, stationary=False, p=None, accuracy=1e-3, maxit=1000, maxit_P=100000):
    r""" Estimate maximum-likelihood HMM

    Generic maximum-likelihood estimation of HMMs

    Parameters
    ----------
    observations : list of numpy arrays representing temporal data
        `observations[i]` is a 1d numpy array corresponding to the observed trajectory index `i`
    n_states : int
        The number of states in the model.
    lag : int
        the lag time at which observations should be read
    initial_model : HMM, optional, default=None
        If specified, the given initial model will be used to initialize the BHMM.
        Otherwise, a heuristic scheme is used to generate an initial guess.
    output : str, optional, default=None
        Output model type from [None, 'gaussian', 'discrete']. If None, will automatically select an output
        model type based on the format of observations.
    reversible : bool, optional, default=True
        If True, a prior that enforces reversible transition matrices (detailed balance) is used;
        otherwise, a standard  non-reversible prior is used.
    stationary : bool, optional, default=False
        If True, the initial distribution of hidden states is self-consistently computed as the stationary
        distribution of the transition matrix. If False, it will be estimated from the starting states.
        Only set this to true if you're sure that the observation trajectories are initiated from a global
        equilibrium distribution.
    p : ndarray (n_states), optional, default=None
        Initial or fixed stationary distribution. If given and stationary=True, transition matrices will be
        estimated with the constraint that they have p as their stationary distribution. If given and
        stationary=False, p is the fixed initial distribution of hidden states.
    accuracy : float
        convergence threshold for EM iteration. When two the likelihood does not increase by more than accuracy, the
        iteration is stopped successfully.
    maxit : int
        stopping criterion for EM iteration. When so many iterations are performed without reaching the requested
        accuracy, the iteration is stopped without convergence (a warning is given)

    Return
    ------
    hmm : :class:`HMM <bhmm.hmm.generic_hmm.HMM>`

    """
    # select output model type
    if output is None:
        output = _guess_output_type(observations)

    if lag > 1:
        observations = lag_observations(observations, lag)

    # construct estimator
    from sktime.markovprocess.bhmm.estimators.maximum_likelihood import MaximumLikelihoodHMM
    est = MaximumLikelihoodHMM(n_states, initial_model=initial_model, output=output,
                               reversible=reversible, stationary=stationary, p=p, accuracy=accuracy,
                               maxit=maxit, maxit_P=maxit_P)
    # run
    est.fit(observations)
    model = est.fetch_model()
    model._lag = lag
    return model


def bayesian_hmm(observations, estimated_hmm, nsample=100, reversible=True, stationary=False,
                 p0_prior='mixed', transition_matrix_prior='mixed', store_hidden=False, callback=None):
    r""" Bayesian HMM based on sampling the posterior

    Generic maximum-likelihood estimation of HMMs

    Parameters
    ----------
    observations : list of numpy arrays representing temporal data
        `observations[i]` is a 1d numpy array corresponding to the observed trajectory index `i`
    estimated_hmm : HMM
        HMM estimated from estimate_hmm or initialize_hmm
    reversible : bool, optional, default=True
        If True, a prior that enforces reversible transition matrices (detailed balance) is used;
        otherwise, a standard  non-reversible prior is used.
    stationary : bool, optional, default=False
        If True, the stationary distribution of the transition matrix will be used as initial distribution.
        Only use True if you are confident that the observation trajectories are started from a global
        equilibrium. If False, the initial distribution will be estimated as usual from the first step
        of the hidden trajectories.
    nsample : int, optional, default=100
        number of Gibbs sampling steps
    p0_prior : None, str, float or ndarray(n)
        Prior for the initial distribution of the HMM. Will only be active
        if stationary=False (stationary=True means that p0 is identical to
        the stationary distribution of the transition matrix).
        Currently implements different versions of the Dirichlet prior that
        is conjugate to the Dirichlet distribution of p0. p0 is sampled from:
        .. math:
            p0 \sim \prod_i (p0)_i^{a_i + n_i - 1}
        where :math:`n_i` are the number of times a hidden trajectory was in
        state :math:`i` at time step 0 and :math:`a_i` is the prior count.
        Following options are available:
        |  'mixed' (default),  :math:`a_i = p_{0,init}`, where :math:`p_{0,init}`
            is the initial distribution of initial_model.
        |  'uniform',  :math:`a_i = 1`
        |  ndarray(n) or float,
            the given array will be used as A.
        |  None,  :math:`a_i = 0`. This option ensures coincidence between
            sample mean an MLE. Will sooner or later lead to sampling problems,
            because as soon as zero trajectories are drawn from a given state,
            the sampler cannot recover and that state will never serve as a starting
            state subsequently. Only recommended in the large data regime and
            when the probability to sample zero trajectories from any state
            is negligible.
    transition_matrix_prior : str or ndarray(n, n)
        Prior for the HMM transition matrix.
        Currently implements Dirichlet priors if reversible=False and reversible
        transition matrix priors as described in [1]_ if reversible=True. For the
        nonreversible case the posterior of transition matrix :math:`P` is:
        .. math:
            P \sim \prod_{i,j} p_{ij}^{b_{ij} + c_{ij} - 1}
        where :math:`c_{ij}` are the number of transitions found for hidden
        trajectories and :math:`b_{ij}` are prior counts.
        |  'mixed' (default),  :math:`b_{ij} = p_{ij,init}`, where :math:`p_{ij,init}`
            is the transition matrix of initial_model. That means one prior
            count will be used per row.
        |  'uniform',  :math:`b_{ij} = 1`
        |  ndarray(n, n) or broadcastable,
            the given array will be used as B.
        |  None,  :math:`b_ij = 0`. This option ensures coincidence between
            sample mean an MLE. Will sooner or later lead to sampling problems,
            because as soon as a transition :math:`ij` will not occur in a
            sample, the sampler cannot recover and that transition will never
            be sampled again. This option is not recommended unless you have
            a small HMM and a lot of data.
    store_hidden : bool, optional, default=False
        store hidden trajectories in sampled HMMs
    callback : function, optional, default=None
        a call back function with no arguments, which if given is being called
        after each computed sample. This is useful for implementing progress bars.

    Return
    ------
    hmm : :class:`SampledHMM <bhmm.hmm.generic_sampled_hmm.SampledHMM>`

    References
    ----------
    .. [1] Trendelkamp-Schroer, B., H. Wu, F. Paul and F. Noe:
        Estimation and uncertainty of reversible Markov models.
        J. Chem. Phys. 143, 174101 (2015).

    """
    from sktime.markovprocess.bhmm.estimators.bayesian_sampling import BayesianHMMSampler
    sampler = BayesianHMMSampler(estimated_hmm.n_states, initial_model=estimated_hmm,
                                 reversible=reversible, stationary=stationary, transition_matrix_sampling_steps=1000,
                                 p0_prior=p0_prior, transition_matrix_prior=transition_matrix_prior,
                                 output=estimated_hmm.output_model.model_type)

    sampler.fit(observations, nsamples=nsample, save_hidden_state_trajectory=store_hidden, callback=callback)

    return sampler
