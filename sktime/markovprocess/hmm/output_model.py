from abc import ABCMeta, abstractmethod
from typing import Optional

import numpy as np

import sktime.markovprocess.hmm._hmm_bindings as _bindings


class OutputModel(object, metaclass=ABCMeta):
    """ HMM output probability model abstract base class. Handles a general output model. """

    def __init__(self, n_hidden_states, n_observable_states, ignore_outliers=True):
        r"""
        Constructs a new output model.

        Parameters
        ----------
        n_hidden_states : int
            The number of output states.
        ignore_outliers : bool
            By outliers we mean observations that have zero probability given the
            current model. ignore_outliers=True means that outliers will be treated
            as if no observation was made, which is equivalent to making this
            observation with equal probability from any hidden state.
            ignore_outliers=True means that an Exception or in the worst case an
            unhandled crash will occur if an outlier is observed.
        """
        self._n_hidden_states = n_hidden_states
        self._n_observable_states = n_observable_states
        self.ignore_outliers = ignore_outliers
        self.found_outliers = False

    @property
    def n_hidden_states(self):
        r""" Number of hidden states """
        return self._n_hidden_states

    @property
    def n_observable_states(self):
        r""" Number of observable states """
        return self._n_observable_states

    @abstractmethod
    def submodel(self, states: np.ndarray):
        r"""
        Restricts this output model to a selection of hidden states.

        Parameters
        ----------
        states : array_like, dtype=int
            hidden states to restrict onto

        Returns
        -------
        Restricted output model.
        """
        pass

    @abstractmethod
    def p_obs(self, obs):
        r"""
        Returns the output probabilities for an entire trajectory and all hidden states

        Parameters
        ----------
        obs : ndarray((T), dtype=int)
            a discrete trajectory of length T

        Returns
        -------
        p_o : ndarray (T,N)
            the probability of generating the symbol at time point t from any of the N hidden states
        """
        pass

    def log_p_obs(self, obs):
        """
        Returns the element-wise logarithm of the output probabilities for an entire trajectory and all hidden states

        This is a default implementation that will take the log of p_obs(obs) and should only be used if p_obs(obs)
        is numerically stable. If there is any danger of running into numerical problems *during* the calculation of
        p_obs, this function should be overwritten in order to compute the log-probabilities directly.

        Parameters
        ----------
        obs : ndarray((T), dtype=int)
            a discrete trajectory of length T

        Return
        ------
        p_o : ndarray (T,N)
            the log probability of generating the symbol at time point t from any of the N hidden states

        """
        return np.log(self.p_obs(obs))

    def _handle_outliers(self, p_o):
        """ Sets observation probabilities of outliers to uniform if ignore_outliers is set.

        Parameters
        ----------
        p_o : ndarray((T, N))
            output probabilities
        """
        if self.ignore_outliers:
            outliers = np.where(p_o.sum(axis=1) == 0)[0]
            if outliers.size > 0:
                p_o[outliers, :] = 1.0
                self.found_outliers = True
        return p_o

    @abstractmethod
    def generate_observation_trajectory(self, s_t):
        """
        Generate synthetic observation data from a given state sequence.

        Parameters
        ----------
        s_t : numpy.array with shape (T,) of int type
            s_t[t] is the hidden state sampled at time t

        Returns
        -------
        o_t : numpy.array with shape (T,) of type dtype
            o_t[t] is the observation associated with state s_t[t]
        """
        pass


class Discrete(OutputModel):
    def __init__(self, output_probability_matrix: np.ndarray, prior: Optional[np.ndarray] = None, ignore_outliers=True):
        if not isinstance(output_probability_matrix, np.ndarray) or output_probability_matrix.ndim != 2:
            raise ValueError("Output probability matrix must be numpy array and two-dimensional.")
        n_hidden = output_probability_matrix.shape[0]
        n_observable = output_probability_matrix.shape[1]
        if prior is not None and (not isinstance(prior, np.ndarray) or prior.ndim != 2
                                  or prior.shape != (n_hidden, n_observable)):
            raise ValueError(f"If prior is given, it must be a numpy array of shape ({n_hidden}, {n_observable}).")
        super().__init__(n_hidden, n_observable, ignore_outliers=ignore_outliers)
        self._output_probability_matrix = output_probability_matrix
        if prior is None:
            prior = np.zeros((n_hidden, n_observable))
        self._prior = prior
