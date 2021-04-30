import numpy as _np
from scipy.sparse import csr_matrix
from scipy.sparse.base import issparse

from deeptime.util.sparse import remove_negative_entries

__docformat__ = "restructuredtext en"
__author__ = "Benjamin Trendelkamp-Schroer, Martin Scherer, Moritz Hoffmann, Frank Noe"
__copyright__ = "Copyright 2014, Computational Molecular Biology Group, FU-Berlin"
__credits__ = ["Benjamin Trendelkamp-Schroer", "Martin Scherer", "Moritz Hoffmann", "Frank Noe"]


# ======================================================================
# Flux matrix operations
# ======================================================================

def flux_matrix(T, pi, qminus, qplus, netflux=True):
    r"""Compute the TPT flux network for the reaction A-->B.

    Parameters
    ----------
    T : (M, M) ndarray
        transition matrix
    pi : (M,) ndarray
        Stationary distribution corresponding to T
    qminus : (M,) ndarray
        Backward comittor
    qplus : (M,) ndarray
        Forward committor
    netflux : boolean
        True: net flux matrix will be computed
        False: gross flux matrix will be computed

    Returns
    -------
    flux : (M, M) ndarray
        Matrix of flux values between pairs of states.

    See also
    --------
    committor.forward_committor, committor.backward_committor

    Notes
    -----
    Computation of the flux network relies on transition path theory
    (TPT) [1]_. Here we use discrete transition path theory [2]_ in
    the transition matrix formulation [3]_.
    The central object used in transition path theory is the
    forward and backward comittor function.

    The TPT (gross) flux is defined as

    .. math::
        f_{ij}=\left \{ \begin{array}{rl}
            \pi_i q_i^{(-)} p_{ij} q_j^{(+)} & i \neq j \\
            0                                & i=j
        \end{array} \right .

    The TPT net flux is then defined as

    .. math:: f_{ij}=\max\{f_{ij} - f_{ji}, 0\} \:\:\:\forall i,j.

    References
    ----------
    .. [1] W. E and E. Vanden-Eijnden.
        Towards a theory of transition paths.
        J. Stat. Phys. 123: 503-523 (2006)
    .. [2] P. Metzner, C. Schuette and E. Vanden-Eijnden.
        Transition Path Theory for Markov Jump Processes.
        Multiscale Model Simul 7: 1192-1219 (2009)
    .. [3] F. Noe, Ch. Schuette, E. Vanden-Eijnden, L. Reich and
        T. Weikl: Constructing the Full Ensemble of Folding Pathways
        from Short Off-Equilibrium Simulations.
        Proc. Natl. Acad. Sci. USA, 106, 19011-19016 (2009)
    """
    if issparse(T):
        flux = T.multiply(qplus[None, ...]).multiply(pi[..., None]).multiply(qminus[..., None])
        # Remove self-fluxes
        flux.setdiag(0)
        flux.eliminate_zeros()
    else:
        flux = pi[:, None] * qminus[:, None] * T * qplus[None, :]
        # Remove self fluxes
        flux[_np.diag_indices(T.shape[0])] = 0.0

    # Return net or gross flux
    if netflux:
        return to_netflux(flux)
    else:
        return flux


def to_netflux(flux):
    r"""Compute the netflux from the gross flux.

    Parameters
    ----------
    flux : (M, M) ndarray
        Matrix of flux values between pairs of states.

    Returns
    -------
    netflux : (M, M) ndarray
        Matrix of netflux values between pairs of states.

    Notes
    -----
    The netflux or effective current is defined as

    .. math::

        f_{ij}^{+}=\max \{ f_{ij}-f_{ji}, 0 \},

    see [1]_.

    :math:`f_{ij}` is the flux for the transition from :math:`A` to
    :math:`B`.

    References
    ----------
    .. [1] P. Metzner, C. Schuette and E. Vanden-Eijnden.
        Transition Path Theory for Markov Jump Processes.
        Multiscale Model Simul 7: 1192-1219 (2009)

    """
    if issparse(flux):
        netflux = remove_negative_entries(flux - flux.T)
    else:
        netflux = _np.maximum(0, flux - flux.T)
    return netflux


def flux_production(F):
    r"""Returns the net flux production for all states

    Parameters
    ----------
    F : (M, M) ndarray
        Matrix of flux values between pairs of states.

    Returns
    -------
    prod : (M,) ndarray
        Array containing flux production (positive) or consumption
        (negative) at each state

    """
    influxes = _np.array(F.sum(axis=0)).flatten()  # all that flows in
    outfluxes = _np.array(F.sum(axis=1)).flatten()  # all that flows out
    prod = outfluxes - influxes  # net flux into nodes
    return prod


def _flux_producers_consumers(F, rtol, atol, producers):
    influxes = _np.array(_np.sum(F, axis=0)).flatten()  # all that flows in
    outfluxes = _np.array(_np.sum(F, axis=1)).flatten()  # all that flows out
    # net out flux absolute
    if producers:
        net_abs = _np.maximum(outfluxes - influxes, 0)
    else:
        net_abs = _np.maximum(influxes - outfluxes, 0)
    # net out flux relative
    prod_rel = net_abs / (_np.maximum(outfluxes, influxes))
    # return all indexes that are produces in terms of absolute and relative tolerance
    return list(_np.where((net_abs > atol) * (prod_rel > rtol))[0])


def flux_producers(F, rtol=1e-05, atol=1e-12):
    r"""Return indexes of states that are net flux producers.

    Parameters
    ----------
    F : (M, M) ndarray
        Matrix of flux values between pairs of states.
    rtol : float
        relative tolerance. fulfilled if max(outflux-influx, 0) / max(outflux,influx) < rtol
    atol : float
        absolute tolerance. fulfilled if max(outflux-influx, 0) < atol

    Returns
    -------
    producers : (M, ) ndarray of int
        indices of states that are net flux producers. May include
        "dirty" producers, i.e.  states that have influx but still
        produce more outflux and thereby violate flux conservation.

    """
    return _flux_producers_consumers(F, rtol, atol, producers=True)


def flux_consumers(F, rtol=1e-05, atol=1e-12):
    r"""Return indexes of states that are net flux producers.

    Parameters
    ----------
    F : (M, M) ndarray
        Matrix of flux values between pairs of states.
    rtol : float
        relative tolerance. fulfilled if max(outflux-influx, 0) / max(outflux,influx) < rtol
    atol : float
        absolute tolerance. fulfilled if max(outflux-influx, 0) < atol

    Returns
    -------
    producers : (M, ) ndarray of int
        indexes of states that are net flux producers. May include
        "dirty" producers, i.e.  states that have influx but still
        produce more outflux and thereby violate flux conservation.

    """
    return _flux_producers_consumers(F, rtol, atol, producers=False)


def coarsegrain(F, sets):
    r"""Coarse-grains the flux to the given sets. See [1]_ .

    Parameters
    ----------
    F : (n, n) ndarray or scipy.sparse matrix
        Matrix of flux values between pairs of states.
    sets : list of array-like of ints
        The sets of states onto which the flux is coarse-grained.

    Notes
    -----
    The coarse grained flux is defined as

    .. math:: fc_{I,J} = \sum_{i \in I,j \in J} f_{i,j}

    Note that if you coarse-grain a net flux, it does n ot necessarily
    have a net flux property anymore. If want to make sure you get a
    netflux, use to_netflux(coarsegrain(F,sets)).

    References
    ----------
    .. [1] F. Noe, Ch. Schuette, E. Vanden-Eijnden, L. Reich and
        T. Weikl: Constructing the Full Ensemble of Folding Pathways
        from Short Off-Equilibrium Simulations.
        Proc. Natl. Acad. Sci. USA, 106, 19011-19016 (2009)

    """
    nnew = len(sets)
    if issparse(F):
        Fc = csr_matrix((nnew, nnew))
        Fin = F.tocsr()
    else:
        Fc = _np.zeros((nnew, nnew))
        Fin = F

    for i in range(0, nnew - 1):
        for j in range(i + 1, nnew):
            I = list(sets[i])
            J = list(sets[j])
            Fc[i, j] = (Fin[I, :][:, J]).sum()
            Fc[j, i] = (Fin[J, :][:, I]).sum()
    return Fc


# ======================================================================
# Total flux, rate and mfpt for the A->B reaction
# ======================================================================


def total_flux(F, A=None):
    r"""Compute the total flux, or turnover flux, that is produced by
        the flux sources and consumed by the flux sinks. [1]_

    Parameters
    ----------
    F : (M, M) ndarray
        Matrix of flux values between pairs of states.
    A : array_like (optional)
        List of integer state labels for set A (reactant)

    Returns
    -------
    F : float
        The total flux, or turnover flux, that is produced by the flux
        sources and consumed by the flux sinks

    References
    ----------
    .. [1] P. Metzner, C. Schuette and E. Vanden-Eijnden.
        Transition Path Theory for Markov Jump Processes.
        Multiscale Model Simul 7: 1192-1219 (2009)

    """
    if A is None:
        prod = flux_production(F)
        outflux = _np.sum(_np.maximum(prod, 0))
        return outflux
    else:
        X = set(_np.arange(F.shape[0]))  # total state space
        A = set(A)
        notA = X.difference(A)

        if issparse(F):
            # Extract rows corresponding to A
            W = F.tocsr()
            W = W[list(A), :]
            # Extract columns corresponding to X\A
            W = W.tocsc()
            W = W[:, list(notA)]

            F = W.sum()
            return F
        else:
            outflux = (F[list(A), :])[:, list(notA)].sum()
            return outflux


def rate(totflux, pi, qminus):
    r"""Transition rate for reaction A to B.

    Parameters
    ----------
    totflux : float
        The total flux between reactant and product
    pi : (M,) ndarray
        Stationary distribution
    qminus : (M,) ndarray
        Backward comittor

    Returns
    -------
    kAB : float
        The reaction rate (per time step of the
        Markov chain)

    See also
    --------
    committor, total_flux, flux_matrix

    Notes
    -----
    Computation of the rate relies on discrete transition path theory
    (TPT). The transition rate, i.e. the total number of reaction events per
    time step, is given in [1]_ as:

    .. math:: k_{AB}=\frac{1}{F} \sum_i \pi_i q_i^{(-)}

    :math:`F` is the total flux for the transition from :math:`A` to
    :math:`B`.

    References
    ----------
    .. [1] F. Noe, Ch. Schuette, E. Vanden-Eijnden, L. Reich and
        T. Weikl: Constructing the Full Ensemble of Folding Pathways
        from Short Off-Equilibrium Simulations.
        Proc. Natl. Acad. Sci. USA, 106, 19011-19016 (2009)

    """
    kAB = totflux / (pi * qminus).sum()
    return kAB


def mfpt(totflux, pi, qminus):
    r"""Mean first passage time for reaction A to B.

    Parameters
    ----------
    totflux : float
        The total flux between reactant and product
    pi : (M,) ndarray
        Stationary distribution
    qminus : (M,) ndarray
        Backward comittor

    Returns
    -------
    tAB : float
        The mean first-passage time for the A to B reaction

    See also
    --------
    rate

    Notes
    -----
    Equal to the inverse rate, see [1]_.

    References
    ----------
    .. [1] F. Noe, Ch. Schuette, E. Vanden-Eijnden, L. Reich and
        T. Weikl: Constructing the Full Ensemble of Folding Pathways
        from Short Off-Equilibrium Simulations.
        Proc. Natl. Acad. Sci. USA, 106, 19011-19016 (2009)

    """
    return 1.0 / rate(totflux, pi, qminus)


###############################################################################
# Pathway decomposition
###############################################################################

def pathways(F, A, B, fraction=1.0, maxiter=1000):
    r"""Decompose flux network into dominant reaction paths. [1]_

    Parameters
    ----------
    F : (M, M) scipy.sparse matrix
        The flux network (matrix of netflux values)
    A : array_like
        The set of starting states
    B : array_like
        The set of end states
    fraction : float, optional
        Fraction of total flux to assemble in pathway decomposition
    maxiter : int, optional
        Maximum number of pathways for decomposition

    Returns
    -------
    paths : list
        List of dominant reaction pathways
    capacities: list
        List of capacities corresponding to each reactions pathway in paths

    Notes
    -----
    The default value for fraction is 1.0, i.e. all dominant reaction
    pathways for the flux network are computed. For large netorks the
    number of possible reaction paths can increase rapidly so that it
    becomes prohibitevely expensive to compute all possible reaction
    paths. To prevent this from happening maxiter sets the maximum
    number of reaction pathways that will be computed.

    For large flux networks it might be necessary to decrease fraction
    or to increase maxiter. It is advisable to begin with a small
    value for fraction and monitor the number of pathways returned
    when increasing the value of fraction.

    References
    ----------
    .. [1] P. Metzner, C. Schuette and E. Vanden-Eijnden.
        Transition Path Theory for Markov Jump Processes.
        Multiscale Model Simul 7: 1192-1219 (2009)

    """
    from .pathways import pathways as impl
    if issparse(F):
        return impl(F, A, B, fraction=fraction, maxiter=maxiter)
    return impl(csr_matrix(F), A, B, fraction=fraction, maxiter=maxiter)
