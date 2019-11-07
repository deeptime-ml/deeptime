import numpy as np

from sktime.base import Model, Estimator
from sktime.numeric import mdot


class PCCA(Model):

    def __init__(self, P=None, pi=None, n_metastable=None,
                 memberships=None, output_probabilities=None, P_coarse=None, pi_coarse=None):
        self._P = P
        self._pi = pi
        self._n_metastable = n_metastable

        self._memberships = memberships
        self._output_probabilities = output_probabilities
        self._P_coarse = P_coarse
        self._pi_coarse = pi_coarse

    @property
    def transition_matrix(self):
        return self._P

    @property
    def stationary_probability(self):
        return self._pi

    @property
    def n_metastable(self):
        return self._n_metastable

    @property
    def memberships(self):
        return self._memberships

    @property
    def output_probabilities(self):
        return self._output_probabilities

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
        return self.memberships.argmax(axis=1) if self.memberships is not None else ()

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
        for i in range(self.n_metastable):
            res.append(np.where(assignment == i)[0])
        return res


class PCCAEstimator(Estimator):
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
    def __init__(self, n_metastable):
        super(PCCAEstimator, self).__init__()
        self.n_metastable = n_metastable

    def _create_model(self) -> PCCA:
        return PCCA()

    # TODO: we can not type annotate this without cyclic import dependencies with MSM...
    def fit(self, msm): # MarkovStateModel):
        from sktime.markovprocess import MarkovStateModel
        if not isinstance(msm, MarkovStateModel):
            raise ValueError(f'msm not of type {type(MarkovStateModel)}, but was {type(msm)}.')

        if msm.is_sparse:
            P = msm.transition_matrix.toarray()
        else:
            P = msm.transition_matrix
        # TODO: can be improved. pcca computes stationary distribution internally, we don't need to compute it twice.
        from msmtools.analysis.dense.pcca import pcca
        memberships = pcca(P, self.n_metastable)

        pi = msm.stationary_distribution

        # coarse-grained stationary distribution
        pi_coarse = np.dot(memberships.T, pi)

        # HMM output matrix
        B = mdot(np.diag(1.0 / pi_coarse), memberships.T, np.diag(pi))
        # renormalize B to make it row-stochastic
        B /= B.sum(axis=1)[:, None]

        # coarse-grained transition matrix
        W = np.linalg.inv(np.dot(memberships.T, memberships))
        A = np.dot(np.dot(memberships.T, P), memberships)
        P_coarse = np.dot(W, A)

        # symmetrize and renormalize to eliminate numerical errors
        X = np.dot(np.diag(pi_coarse), P_coarse)
        P_coarse = X / X.sum(axis=1)[:, None]

        self._model.__init__(P=P, pi=msm.stationary_distribution, n_metastable=self.n_metastable,
                             memberships=memberships, output_probabilities=B,
                             P_coarse=P_coarse, pi_coarse=pi_coarse)
        return self
