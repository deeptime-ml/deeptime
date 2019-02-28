import warnings

import numpy as np

from sktime.base import Model


class PCCA(Model):

    """
    PCCA+ spectral clustering method with optimized memberships [1]_
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
    [2] F. Noe, multiset PCCA and HMMs, in preparation.
    [3] F. Noe, H. Wu, J.-H. Prinz and N. Plattner:
        Projected and hidden Markov models for calculating kinetics and metastable states of complex molecules
        J. Chem. Phys. 139, 184114 (2013)
    """
    def __init__(self, P, m):
        # remember input
        from scipy.sparse import issparse
        if issparse(P):
            warnings.warn('PCCA is only implemented for dense matrices, '
                          'converting sparse transition matrix to dense ndarray.')
            P = P.toarray()
        self.P = P
        self.m = m
        self._coarse_grain(self.P, self.m)

    def _coarse_grain(self, P, m):
        # pcca coarse-graining
        # --------------------
        # PCCA memberships
        # TODO: can be improved. pcca computes stationary distribution internally, we don't need to compute it twice.
        from msmtools.analysis.dense.pcca import pcca
        self._M = pcca(P, m)

        # stationary distribution
        # TODO: in msmtools we recomputed this from P, we actually want to use pi from the msm obj, but this caused #1208
        from msmtools.analysis import stationary_distribution
        self._pi = stationary_distribution(P)

        # coarse-grained stationary distribution
        self._pi_coarse = np.dot(self._M.T, self._pi)

        # HMM output matrix
        self._B = np.dot(np.dot(np.diag(1.0 / self._pi_coarse), self._M.T), np.diag(self._pi))
        # renormalize B to make it row-stochastic
        self._B /= self._B.sum(axis=1)[:, None]
        self._B /= self._B.sum(axis=1)[:, None]

        # coarse-grained transition matrix
        W = np.linalg.inv(np.dot(self._M.T, self._M))
        A = np.dot(np.dot(self._M.T, P), self._M)
        self._P_coarse = np.dot(W, A)

        # symmetrize and renormalize to eliminate numerical errors
        X = np.dot(np.diag(self._pi_coarse), self._P_coarse)
        self._P_coarse = X / X.sum(axis=1)[:, None]

    @property
    def transition_matrix(self):
        return self.P

    @property
    def stationary_probability(self):
        return self._pi

    @property
    def n_metastable(self):
        return self.m

    @property
    def memberships(self):
        return self._M

    @property
    def output_probabilities(self):
        return self._B

    @property
    def coarse_grained_transition_matrix(self):
        return self._P_coarse

    @property
    def coarse_grained_stationary_probability(self):
        return self._pi_coarse

    @property
    def metastable_assignment(self):
        """
        Crisp clustering using PCCA. This is only recommended for visualization purposes. You *cannot* compute any
        actual quantity of the coarse-grained kinetics without employing the fuzzy memberships!
        Returns
        -------
        For each microstate, the metastable state it is located in.
        """
        return np.argmax(self.memberships, axis=1)

    @property
    def metastable_sets(self):
        """
        Crisp clustering using PCCA. This is only recommended for visualization purposes. You *cannot* compute any
        actual quantity of the coarse-grained kinetics without employing the fuzzy memberships!
        Returns
        -------
        A list of length equal to metastable states. Each element is an array with microstate indexes contained in it
        """
        res = []
        assignment = self.metastable_assignment
        for i in range(self.m):
            res.append(np.where(assignment == i)[0])
        return res
