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

from abc import ABCMeta, abstractmethod

import numpy as np


class OutputModel(object, metaclass=ABCMeta):
    """ HMM output probability model abstract base class. Handles a general output model.

    Parameters
    ----------
    n_states : int
        The number of output states.
    ignore_outliers : bool
        By outliers we mean observations that have zero probability given the
        current model. ignore_outliers=True means that outliers will be treated
        as if no observation was made, which is equivalent to making this
        observation with equal probability from any hidden state.
        ignore_outliers=True means that an Exception or in the worst case an
        unhandled crash will occur if an outlier is observed.
        If outliers have been found, the flag found_outliers will be set True

    """

    def __init__(self, n_states, ignore_outliers=True):
        self._n_states = n_states
        self.ignore_outliers = ignore_outliers
        self.found_outliers = False

    @property
    def n_states(self):
        r""" Number of hidden states """
        return self._n_states

    @abstractmethod
    def sub_output_model(self, states):
        """ Returns output model on a subset of states """
        pass

    @abstractmethod
    def p_obs(self, obs, out=None):
        pass

    def log_p_obs(self, obs, out=None):
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
        if out is None:
            return np.log(self.p_obs(obs))
        else:
            self.p_obs(obs, out=out)
            np.log(out, out=out)
            return out

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
