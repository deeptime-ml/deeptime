import warnings

import numpy as np

from sktime.base import Model
from sktime.numeric import mdot


# TODO: should pass pi to msmtools once it's supported.
def pcca(P, m):
    """PCCA+ spectral clustering method with optimized memberships [1]_

    Clusters the first m eigenvectors of a transition matrix in order to cluster the states.
    This function does not assume that the transition matrix is fully connected. Disconnected sets
    will automatically define the first metastable states, with perfect membership assignments.

    Parameters
    ----------
    P : ndarray (n,n)
        Transition matrix.

    m : int
        Number of clusters to group to.

    References
    ----------
    [1] S. Roeblitz and M. Weber, Fuzzy spectral clustering by PCCA+:
        application to Markov state models and data classification.
        Adv Data Anal Classif 7, 147-179 (2013).
    """
    assert 0 < m <= P.shape[0]
    from scipy.sparse import issparse
    if issparse(P):
        warnings.warn('PCCA is only implemented for dense matrices, '
                      'converting sparse transition matrix to dense ndarray.', stacklevel=2)
        P = P.toarray()
    # memberships
    # TODO: can be improved. pcca computes stationary distribution internally, we don't need to compute it twice.
    from msmtools.analysis.dense.pcca import pcca as _algorithm_impl
    M = _algorithm_impl(P, m)

    # stationary distribution
    # TODO: in msmtools we recomputed this from P, we actually want to use pi from the msm obj, but this caused #1208
    from msmtools.analysis import stationary_distribution
    pi = stationary_distribution(P)

    # coarse-grained stationary distribution
    pi_coarse = np.dot(M.T, pi)

    # HMM output matrix
    B = mdot(np.diag(1.0 / pi_coarse), M.T, np.diag(pi))
    # renormalize B to make it row-stochastic
    B /= B.sum(axis=1)[:, None]

    # coarse-grained transition matrix
    W = np.linalg.inv(np.dot(M.T, M))
    A = np.dot(np.dot(M.T, P), M)
    P_coarse = np.dot(W, A)

    # symmetrize and renormalize to eliminate numerical errors
    X = np.dot(np.diag(pi_coarse), P_coarse)
    P_coarse = X / X.sum(axis=1)[:, None]

    return PCCAModel(P_coarse, pi_coarse, M, B)


class PCCAModel(Model):
    """
    Model for PCCA+ spectral clustering method with optimized memberships [1]_
    Clusters the first m eigenvectors of a transition matrix in order to cluster the states.
    This function does not assume that the transition matrix is fully connected. Disconnected sets
    will automatically define the first metastable states, with perfect membership assignments.

    Parameters
    ----------
    P_coarse : ndarray (n,n)
        Coarse transition matrix.
    pi_coarse : ndarray (n,)
        Coarse stationary distribution
    memberships : ndarray (n,m)
        The pcca memberships to clusters
    metastable_distributions : ndarray (m, n)
        metastable distributions

    References
    ----------
    [1] S. Roeblitz and M. Weber, Fuzzy spectral clustering by PCCA+:
        application to Markov state models and data classification.
        Adv Data Anal Classif 7, 147-179 (2013).
    [2] F. Noe, H. Wu, J.-H. Prinz and N. Plattner:
        Projected and hidden Markov models for calculating kinetics and metastable states of complex molecules
        J. Chem. Phys. 139, 184114 (2013)
    """

    def __init__(self, P_coarse, pi_coarse, memberships, metastable_distributions):
        self._P_coarse = P_coarse
        self._pi_coarse = pi_coarse
        self._memberships = memberships
        self._metastable_distributions = metastable_distributions
        self.m = self._memberships.shape[1]

    @property
    def n_metastable(self):
        return self.m

    @property
    def memberships(self):
        r""" Probabilities of MarkovStateModel states to belong to a metastable state by PCCA+

        Returns the memberships of active set states to metastable sets.

        Returns
        -------
        M : ndarray((n,m))
            A matrix containing the probability or membership of each state to be
            assigned to each metastable set, i.e. p(metastable | state).
            The row sums of M are 1.
        """
        return self._memberships

    @property
    def distributions(self):
        r""" Probability of metastable states to visit an MarkovStateModel state by PCCA+

        Returns the probability distributions of active set states within
        each metastable set by combining the PCCA+ method with
        Bayesian inversion as described in [2]_.

        Returns
        -------
        p_out : ndarray (m,n)
            A matrix containing the probability distribution of each active set
            state, given that we are in one of the m metastable sets,
            i.e. p(state | metastable). The row sums of p_out are 1.
        """
        return self._metastable_distributions

    output_probabilities = distributions

    @property
    def coarse_grained_transition_matrix(self):
        return self._P_coarse

    @property
    def coarse_grained_stationary_probability(self):
        return self._pi_coarse

    @property
    def assignments(self):
        """ Assignment of states to metastable sets using PCCA++

        Computes the assignment to metastable sets for active set states using
        the PCCA++ method [1]_.

        This is only recommended for visualization purposes. You *cannot* compute
        any actual quantity of the coarse-grained kinetics without employing the
        fuzzy memberships!

        Returns
        -------
        assignments : ndarray (n,)
            For each MarkovStateModel state, the metastable state it is located in.

        """
        return np.argmax(self.memberships, axis=1)

    @property
    def sets(self):
        """ Metastable sets using PCCA+

        Computes the metastable sets of active set states within each
        metastable set using the PCCA+ method [1]_.

        This is only recommended for visualization purposes. You *cannot*
        compute any actual quantity of the coarse-grained kinetics without
        employing the fuzzy memberships!

        Returns
        -------
        sets : list of ndarray
            A list of length equal to metastable states. Each element is an
            array with microstate indexes contained in it
        """
        res = []
        assignment = self.assignments
        for i in range(self.m):
            res.append(np.where(assignment == i)[0])
        return res
