"""This module implements the transition matrix functionality"""

import numpy as np


def log_likelihood(C, T):
    """
        implementation of likelihood of C given T
    """
    C = C.tocsr()
    T = T.tocsr()
    ind = np.nonzero(C)
    relT = np.array(T[ind])[0, :]
    relT = np.log(relT)
    relC = np.array(C[ind])[0, :]

    return relT.dot(relC)
