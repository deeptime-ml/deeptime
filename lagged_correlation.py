from __future__ import absolute_import
from __future__ import print_function
__author__ = 'noe'

import numpy as np

class LaggedCorrelation(object):

    def __init__(self, output_dimension, tau=1):
        """ Computes correlation matrices C0 and Ctau from a bunch of trajectories

        Parameters
        ----------
        output_dimension: int
            Number of basis functions.
        tau: int
            Lag time

        """
        self.tau = tau
        self.output_dimension = output_dimension
        # Initialize the two correlation matrices:
        self.Ct = np.zeros((self.output_dimension, self.output_dimension))
        self.C0 = np.zeros((self.output_dimension, self.output_dimension))
        # Create counter for the frames used for C0, Ct:
        self.nC0 = 0
        self.nCt = 0

    def add(self, X):
        """ Adds trajectory to the running estimate for computing mu, C0 and Ct:

        Parameters
        ----------
        X: ndarray (T,N)
            basis function trajectory of T time steps for N basis functions.

        """
        # Raise an error if output dimension is wrong:
        if not (X.shape[1] == self.output_dimension):
            raise Exception("Number of basis functions is incorrect."+
                            "Got %d, expected %d."%(X.shape[1],
                                                    self.output_dimension))
        # Print message if number of time steps is too small:
        if X.shape[0] <= self.tau:
            raise ValueError("Number of time steps is too small.")

        # Get the time-lagged data:
        Y1 = X[self.tau:,:]
        # Remove the last tau frames from X:
        Y2 = X[:-self.tau,:]
        # Get the number of time steps in this trajectory:
        TX = 1.0*Y1.shape[0]
        # Update time-lagged correlation matrix:
        self.Ct += np.dot(Y1.T,Y2)
        self.nCt += TX
        # Update the instantaneous correlation matrix:
        self.C0 += np.dot(Y1.T,Y1) + np.dot(Y2.T,Y2)
        self.nC0 += 2*TX

    def GetC0(self):
        """ Returns the current estimate of C0:

        Returns
        -------
        C0: ndarray (N,N)
            time instantaneous correlation matrix of N basis function.

        """
        return 0.5*(self.C0 + self.C0.T)/(self.nC0 - 1)

    def GetCt(self):
        """ Returns the current estimate of Ctau

        Returns
        -------
        Ct: ndarray (N,N)
            time lagged correlation matrix of N basis function.

        """
        return 0.5*(self.Ct + self.Ct.T)/(self.nCt - 1)
