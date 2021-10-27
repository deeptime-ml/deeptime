"""
@author: Susanna Roeblitz, Marcus Weber, Frank Noe

modified from ZIBMolPy which can also be found on Github:
https://github.com/CMD-at-ZIB/ZIBMolPy/blob/master/ZIBMolPy_package/ZIBMolPy/algorithms.py
"""

import warnings
import numpy as np
from scipy.sparse import issparse
import math


def _pcca_connected_isa(eigenvectors, n_clusters):
    """
    PCCA+ spectral clustering method using the inner simplex algorithm.

    Clusters the first n_cluster eigenvectors of a transition matrix in order to cluster the states.
    This function assumes that the state space is fully connected, i.e. the transition matrix whose
    eigenvectors are used is supposed to have only one eigenvalue 1, and the corresponding first
    eigenvector (evec[:,0]) must be constant.

    Parameters
    ----------
    eigenvectors : ndarray
        A matrix with the sorted eigenvectors in the columns. The stationary eigenvector should
        be first, then the one to the slowest relaxation process, etc.

    n_clusters : int
        Number of clusters to group to.

    Returns
    -------
    chi : ndarray (n x m)
        A matrix containing the probability or membership of each state to be assigned to each cluster.
        The rows sum to 1.

    rot_mat : ndarray (m x m)
        A rotation matrix that rotates the dominant eigenvectors to yield the PCCA memberships, i.e.:
        chi = np.dot(evec, rot_matrix

    References
    ----------
    [1] P. Deuflhard and M. Weber, Robust Perron cluster analysis in conformation dynamics.
        in: Linear Algebra Appl. 398C M. Dellnitz and S. Kirkland and M. Neumann and C. Schuette (Editors)
        Elsevier, New York, 2005, pp. 161-184

    """
    (n, m) = eigenvectors.shape

    # do we have enough eigenvectors?
    if n_clusters > m:
        raise ValueError("Cannot cluster the (" + str(n) + " x " + str(m)
                         + " eigenvector matrix to " + str(n_clusters) + " clusters.")

    # check if the first, and only the first eigenvector is constant
    diffs = np.abs(np.max(eigenvectors, axis=0) - np.min(eigenvectors, axis=0))
    assert diffs[0] < 1e-6, "First eigenvector is not constant. This indicates that the transition matrix " \
                            "is not connected or the eigenvectors are incorrectly sorted. Cannot do PCCA."
    assert diffs[1] > 1e-6, "An eigenvector after the first one is constant. " \
                            "Probably the eigenvectors are incorrectly sorted. Cannot do PCCA."

    # local copy of the eigenvectors
    c = eigenvectors[:, list(range(n_clusters))]

    ortho_sys = np.copy(c)
    max_dist = 0.0

    # representative states
    ind = np.zeros(n_clusters, dtype=np.int32)

    # select the first representative as the most outlying point
    for (i, row) in enumerate(c):
        if np.linalg.norm(row, 2) > max_dist:
            max_dist = np.linalg.norm(row, 2)
            ind[0] = i

    # translate coordinates to make the first representative the origin
    ortho_sys -= c[ind[0], None]

    # select the other m-1 representatives using a Gram-Schmidt orthogonalization
    for k in range(1, n_clusters):
        max_dist = 0.0
        temp = np.copy(ortho_sys[ind[k - 1]])

        # select next farthest point that is not yet a representative
        for (i, row) in enumerate(ortho_sys):
            row -= np.dot(np.dot(temp, np.transpose(row)), temp)
            distt = np.linalg.norm(row, 2)
            if distt > max_dist and i not in ind[0:k]:
                max_dist = distt
                ind[k] = i
        ortho_sys /= np.linalg.norm(ortho_sys[ind[k]], 2)

    # print "Final selection ", ind

    # obtain transformation matrix of eigenvectors to membership matrix
    rot_mat = np.linalg.inv(c[ind])
    # print "Rotation matrix \n ", rot_mat

    # compute membership matrix
    chi = np.dot(c, rot_mat)
    # print "chi \n ", chi

    return chi, rot_mat


def _opt_soft(eigenvectors, rot_matrix, n_clusters):
    """
    Optimizes the PCCA+ rotation matrix such that the memberships are exclusively nonnegative.

    Parameters
    ----------
    eigenvectors : ndarray
        A matrix with the sorted eigenvectors in the columns. The stationary eigenvector should
        be first, then the one to the slowest relaxation process, etc.

    rot_matrix : ndarray (m x m)
        nonoptimized rotation matrix

    n_clusters : int
        Number of clusters to group to.

    Returns
    -------
    rot_mat : ndarray (m x m)
        Optimized rotation matrix that rotates the dominant eigenvectors to yield the PCCA memberships, i.e.:
        chi = np.dot(evec, rot_matrix

    References
    ----------
    [1] S. Roeblitz and M. Weber, Fuzzy spectral clustering by PCCA+:
        application to Markov state models and data classification.
        Adv Data Anal Classif 7, 147-179 (2013).

    """
    # only consider first n_clusters eigenvectors
    eigenvectors = eigenvectors[:, :n_clusters]

    # crop first row and first column from rot_matrix
    # rot_crop_matrix = rot_matrix[1:,1:]
    rot_crop_matrix = rot_matrix[1:][:, 1:]

    (x, y) = rot_crop_matrix.shape

    # reshape rot_crop_matrix into linear vector
    rot_crop_vec = np.reshape(rot_crop_matrix, x * y)

    # Susanna Roeblitz' target function for optimization
    def susanna_func(rot_crop_vec, eigvectors):
        # reshape into matrix
        rot_crop_matrix = np.reshape(rot_crop_vec, (x, y))
        # fill matrix
        rot_matrix = _fill_matrix(rot_crop_matrix, eigvectors)

        result = 0
        for i in range(0, n_clusters):
            for j in range(0, n_clusters):
                result += np.power(rot_matrix[j, i], 2) / rot_matrix[0, i]
        return -result

    from scipy.optimize import fmin

    rot_crop_vec_opt = fmin(susanna_func, rot_crop_vec, args=(eigenvectors,), disp=False)

    rot_crop_matrix = np.reshape(rot_crop_vec_opt, (x, y))
    rot_matrix = _fill_matrix(rot_crop_matrix, eigenvectors)

    return rot_matrix


def _fill_matrix(rot_crop_matrix, eigvectors):
    """ Helper function for opt_soft """

    (x, y) = rot_crop_matrix.shape

    row_sums = np.sum(rot_crop_matrix, axis=1)
    row_sums = np.reshape(row_sums, (x, 1))

    # add -row_sums as leftmost column to rot_crop_matrix
    rot_crop_matrix = np.concatenate((-row_sums, rot_crop_matrix), axis=1)

    tmp = -np.dot(eigvectors[:, 1:], rot_crop_matrix)

    tmp_col_max = np.max(tmp, axis=0)
    tmp_col_max = np.reshape(tmp_col_max, (1, y + 1))

    tmp_col_max_sum = np.sum(tmp_col_max)

    # add col_max as top row to rot_crop_matrix and normalize
    rot_matrix = np.concatenate((tmp_col_max, rot_crop_matrix), axis=0)
    rot_matrix /= tmp_col_max_sum

    return rot_matrix


def _pcca_connected(P, n, pi=None):
    r"""PCCA+ spectral clustering method with optimized memberships [1]_

    Clusters the first n_cluster eigenvectors of a transition matrix in order to cluster the states.
    This function assumes that the transition matrix is fully connected.

    Parameters
    ----------
    P : ndarray (n,n)
        Transition matrix.
    n : int
        Number of clusters to group to.
    pi: ndarray(n,), optional, default=None
        Stationary distribution if available.

    Returns
    -------
    chi : ndarray (n x m)
        A matrix containing the probability or membership of each state to be assigned to each cluster.
        The rows sum to 1.

    References
    ----------
    [1] S. Roeblitz and M. Weber, Fuzzy spectral clustering by PCCA+:
        application to Markov state models and data classification.
        Adv Data Anal Classif 7, 147-179 (2013).
    """

    # test connectivity
    from deeptime.markov.tools.estimation import connected_sets

    labels = connected_sets(P)
    n_components = len(labels)  # (n_components, labels) = connected_components(P, connection='strong')
    if n_components > 1:
        raise ValueError("Transition matrix is disconnected. Cannot use pcca_connected.")

    if pi is None:
        from deeptime.markov.tools.analysis import stationary_distribution
        pi = stationary_distribution(P)
    else:
        if pi.shape[0] != P.shape[0]:
            raise ValueError(f"Stationary distribution must span entire state space but got {pi.shape[0]} states "
                             f"instead of {P.shape[0]}.")
        pi /= pi.sum()  # make sure it is normalized

    from deeptime.markov.tools.analysis import is_reversible

    if not is_reversible(P, mu=pi):
        raise ValueError("Transition matrix does not fulfill detailed balance. "
                         "Make sure to call pcca with a reversible transition matrix estimate")
    # TODO: Susanna mentioned that she has a potential fix for nonreversible matrices by replacing each complex conjugate
    #      pair by the real and imaginary components of one of the two vectors. We could use this but would then need to
    #      orthonormalize all eigenvectors e.g. using Gram-Schmidt orthonormalization. Currently there is no theoretical
    #      foundation for this, so I'll skip it for now.

    # right eigenvectors, ordered
    from deeptime.markov.tools.analysis import eigenvectors

    evecs = eigenvectors(P, n)

    # orthonormalize
    for i in range(n):
        evecs[:, i] /= math.sqrt(np.dot(evecs[:, i] * pi, evecs[:, i]))
    # make first eigenvector positive
    evecs[:, 0] = np.abs(evecs[:, 0])

    # Is there a significant complex component?
    if not np.alltrue(np.isreal(evecs)):
        warnings.warn(
            "The given transition matrix has complex eigenvectors, so it doesn't exactly fulfill detailed balance. "
            "Forcing eigenvectors to be real and continuing. Be aware that this is not theoretically solid.")
    evecs = np.real(evecs)

    # create initial solution using PCCA+. This could have negative memberships
    chi, rot_matrix = _pcca_connected_isa(evecs, n)

    # optimize the rotation matrix with PCCA++.
    rot_matrix = _opt_soft(evecs, rot_matrix, n)

    # These memberships should be nonnegative
    memberships = np.dot(evecs[:, :], rot_matrix)

    # We might still have numerical errors. Force memberships to be in [0,1]
    memberships = np.clip(memberships, 0., 1.)

    for i in range(0, np.shape(memberships)[0]):
        memberships[i] /= np.sum(memberships[i])

    return memberships


def pcca(P: np.ndarray, m: int, pi: np.ndarray = None, transition_matrix_tol: float = 1e-12):
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
    pi : ndarray(n,), optional, default=None
        Stationary distribution if available. Should be defined piecewise over the connected sets.
    transition_matrix_tol : float, optional, default=1e-12
        Tolerance under which P is checked to be a transition matrix.

    Returns
    -------
    chi : ndarray (n x m)
        A matrix containing the probability or membership of each state to be assigned to each cluster.
        The rows sum to 1.

    References
    ----------
    [1] S. Roeblitz and M. Weber, Fuzzy spectral clustering by PCCA+:
        application to Markov state models and data classification.
        Adv Data Anal Classif 7, 147-179 (2013).
    [2] F. Noe, multiset PCCA and HMMs, in preparation.
    """
    # imports
    from deeptime.markov.tools.estimation import connected_sets
    from deeptime.markov.tools.analysis import eigenvalues, is_transition_matrix, hitting_probability

    # validate input
    n = np.shape(P)[0]
    if m > n:
        raise ValueError(f"Number of metastable states m={m} exceeds number of states of transition matrix n={n}")
    if not is_transition_matrix(P, tol=transition_matrix_tol):
        raise ValueError("Input matrix is not a transition matrix.")
    if pi is not None and pi.ndim > 1:
        raise ValueError("Stationary distribution must be given as one-dimensional array or left None.")
    if pi is not None and pi.shape[0] != n:
        raise ValueError(f"Stationary distribution must be defined on entire space, piecewise if the transition matrix "
                         f"has multiple connected components. It covered {pi.shape[0]} != {n} states.")

    # prepare output
    chi = np.zeros((n, m))

    # test connectivity
    components = connected_sets(P)
    n_components = len(components)  # (n_components, labels) = connected_components(P, connection='strong')

    # store components as closed (with positive equilibrium distribution)
    # or as transition states (with vanishing equilibrium distribution)
    closed_components = []
    transition_states = []
    for i in range(n_components):
        component = components[i]  # np.argwhere(labels==i).flatten()
        rest = list(set(range(n)) - set(component))
        # is component closed?
        if np.sum(P[component, :][:, rest]) == 0:
            closed_components.append(component)
        else:
            transition_states.append(component)
    n_closed_components = len(closed_components)
    closed_states = np.concatenate(closed_components)
    if len(transition_states) == 0:
        transition_states = np.array([], dtype=int)
    else:
        transition_states = np.concatenate(transition_states)

    # check if we have enough clusters to support the disconnected sets
    if m < len(closed_components):
        raise ValueError(f"Number of metastable states m={m} is too small. Transition matrix "
                         f"has {len(closed_components)} disconnected components.")

    # We collect eigenvalues in order to decide which
    closed_components_Psub = []
    closed_components_ev = []
    closed_components_enum = []
    for i in range(n_closed_components):
        component = closed_components[i]

        # compute eigenvalues in submatrix
        Psub = P[component, :][:, component]
        closed_components_Psub.append(Psub)
        closed_components_ev.append(eigenvalues(Psub))
        closed_components_enum.append(i * np.ones((component.size,), dtype=int))

    # flatten
    closed_components_ev_flat = np.hstack(closed_components_ev)
    closed_components_enum_flat = np.hstack(closed_components_enum)
    # which components should be clustered?
    component_indexes = closed_components_enum_flat[np.argsort(closed_components_ev_flat)][0:m]
    # cluster each component
    ipcca = 0
    for i in range(n_closed_components):
        component = closed_components[i]
        # how many PCCA states in this component?
        m_by_component = np.shape(np.argwhere(component_indexes == i))[0]

        # if 1, then the result is trivial
        if m_by_component == 1:
            chi[component, ipcca] = 1.0
            ipcca += 1
        elif m_by_component > 1:
            # print "submatrix: ",closed_components_Psub[i]
            chi[component, ipcca:ipcca + m_by_component] = _pcca_connected(
                closed_components_Psub[i], m_by_component, pi=None if pi is None else pi[closed_components[i]]
            )
            ipcca += m_by_component
        else:
            raise RuntimeError(f"Component {i} spuriously has {m_by_component} pcca sets")

    # finally assign all transition states
    if transition_states.size > 0:
        # make all closed states absorbing, so we can see which closed state we hit first
        Pabs = P.copy()
        Pabs[closed_states, :] = 0.0
        Pabs[closed_states, closed_states] = 1.0
        for i in range(closed_states.size):
            # hitting probability to each closed state
            h = hitting_probability(Pabs, closed_states[i])
            for j in range(transition_states.size):
                # transition states belong to closed states with the hitting probability, and inherit their chi
                chi[transition_states[j]] += h[transition_states[j]] * chi[closed_states[i]]

    # check if we have m metastable sets. If less than m, we must raise
    nmeta = np.count_nonzero(chi.sum(axis=0))
    if nmeta < m:
        raise RuntimeError(f"{m} metastable states requested, but transition matrix only has {nmeta}. "
                           f"Consider using a prior or request less metastable states.")

    return chi


def coarsegrain(P, n):
    """
    Coarse-grains transition matrix P to n sets using PCCA

    Coarse-grains transition matrix P such that the dominant eigenvalues are preserved, using:

    ..math:
        \tilde{P} = M^T P M (M^T M)^{-1}

    See [2]_ for the derivation of this form from the coarse-graining method first derived in [1]_.

    References
    ----------
    [1] S. Kube and M. Weber
        A coarse graining method for the identification of transition rates between molecular conformations.
        J. Chem. Phys. 126, 024103 (2007)
    [2] F. Noe, H. Wu, J.-H. Prinz and N. Plattner:
        Projected and hidden Markov models for calculating kinetics and metastable states of complex molecules
        J. Chem. Phys. 139, 184114 (2013)
    """
    M = pcca(P, n)
    # coarse-grained transition matrix
    W = np.linalg.inv(np.dot(M.T, M))
    A = np.dot(np.dot(M.T, P), M)
    P_coarse = np.dot(W, A)

    # symmetrize and renormalize to eliminate numerical errors
    from deeptime.markov.tools.analysis import stationary_distribution
    pi_coarse = np.dot(M.T, stationary_distribution(P))
    X = np.dot(np.diag(pi_coarse), P_coarse)
    P_coarse = X / X.sum(axis=1)[:, None]

    return P_coarse


class PCCA:
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
        # TODO: can be improved: if we have eigendecomposition already, this can be exploited.
        # remember input
        if issparse(P):
            warnings.warn('pcca is only implemented for dense matrices, '
                          'converting sparse transition matrix to dense ndarray.')
            P = P.toarray()
        self.P = P
        self.m = m

        # pcca coarse-graining
        # --------------------
        # PCCA memberships
        # TODO: can be improved. pcca computes stationary distribution internally, we don't need to compute it twice.
        self._M = pcca(P, m)

        # stationary distribution
        from deeptime.markov.tools.analysis import stationary_distribution as _sd

        self._pi = _sd(P)

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
