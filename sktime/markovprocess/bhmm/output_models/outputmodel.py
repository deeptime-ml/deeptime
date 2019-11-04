
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
from abc import ABCMeta, abstractmethod


class OutputModel(object):
    """
    HMM output probability model abstract base class.

    """

    # Abstract base class.
    __metaclass__ = ABCMeta

    # implementation codes
    __IMPL_PYTHON__ = 0
    __IMPL_C__ = 1

    # implementation used
    __impl__ = __IMPL_PYTHON__

    def __init__(self, nstates, ignore_outliers=True):
        """
        Create a general output model.

        Parameters
        ----------
        nstates : int
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
        self._nstates = nstates
        self.ignore_outliers = ignore_outliers
        self.found_outliers = False

        return

    @property
    def nstates(self):
        r""" Number of hidden states """
        return self._nstates

    def set_implementation(self, impl):
        """
        Sets the implementation of this module

        Parameters
        ----------
        impl : str
            One of ["python", "c"]

        """
        if impl.lower() == 'python':
            self.__impl__ = self.__IMPL_PYTHON__
        elif impl.lower() == 'c':
            self.__impl__ = self.__IMPL_C__
        else:
            import warnings
            warnings.warn('Implementation '+impl+' is not known. Using the fallback python implementation.')
            self.__impl__ = self.__IMPL_PYTHON__

    @abstractmethod
    def sub_output_model(self, states):
        """ Returns output model on a subset of states """
        pass

    def log_p_obs(self, obs, out=None, dtype=np.float32):
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
            self.p_obs(obs, out=out, dtype=dtype)
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
            outliers = np.where(p_o.sum(axis=1)==0)[0]
            if outliers.size > 0:
                p_o[outliers, :] = 1.0
                self.found_outliers = True
        return p_o

    @abstractmethod
    def generate_observation_trajectory(self, s_t, dtype=None):
        """
        Generate synthetic observation data from a given state sequence.

        Parameters
        ----------
        s_t : numpy.array with shape (T,) of int type
            s_t[t] is the hidden state sampled at time t
        dtype : numpy.dtype, optional, default=None
            The datatype to return the resulting observations in.

        Returns
        -------
        o_t : numpy.array with shape (T,) of type dtype
            o_t[t] is the observation associated with state s_t[t]
        """
        pass
