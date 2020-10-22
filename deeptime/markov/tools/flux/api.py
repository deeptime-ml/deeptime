r"""
===================
 Flux API
===================

"""

from scipy.sparse.base import issparse
from scipy.sparse.sputils import isdense
from scipy.sparse import csr_matrix

from . import dense
from . import sparse

__docformat__ = "restructuredtext en"
__author__ = "Benjamin Trendelkamp-Schroer, Martin Scherer, Frank Noe"
__copyright__ = "Copyright 2014, Computational Molecular Biology Group, FU-Berlin"
__credits__ = ["Benjamin Trendelkamp-Schroer", "Martin Scherer", "Frank Noe"]

__version__ = "2.0.0"
__maintainer__ = "Martin Scherer"
__email__ = "m.scherer AT fu-berlin DOT de"

__all__ = ['flux_matrix',
           'to_netflux',
           'flux_production',
           'flux_producers',
           'flux_consumers',
           'coarsegrain',
           'total_flux',
           'rate',
           'mfpt',
           'pathways']

_type_not_supported = \
    TypeError("T is not a numpy.ndarray or a scipy.sparse matrix.")


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

    Notes
    -----
    Computation of the flux network relies on transition path theory
    (TPT) [1]_. Here we use discrete transition path theory [2]_ in
    the transition matrix formulation [3]_.

    See also
    --------
    committor.forward_committor, committor.backward_committor

    Notes
    -----
    Computation of the flux network relies on transition path theory
    (TPT). The central object used in transition path theory is the
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
        return sparse.tpt.flux_matrix(T, pi, qminus, qplus, netflux=netflux)
    elif isdense(T):
        return dense.tpt.flux_matrix(T, pi, qminus, qplus, netflux=netflux)
    else:
        raise _type_not_supported


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
        return sparse.tpt.to_netflux(flux)
    elif isdense(flux):
        return dense.tpt.to_netflux(flux)
    else:
        raise _type_not_supported


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
    return dense.tpt.flux_production(F)  # works for dense or sparse


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
        indexes of states that are net flux producers. May include
        "dirty" producers, i.e.  states that have influx but still
        produce more outflux and thereby violate flux conservation.

    """
    # works for dense or sparse
    return dense.tpt.flux_producers(F, rtol=rtol, atol=atol)


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
    # works for dense or sparse
    return dense.tpt.flux_consumers(F, rtol=rtol, atol=atol)


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
    if issparse(F):
        return sparse.tpt.coarsegrain(F, sets)
    elif isdense(F):
        return dense.tpt.coarsegrain(F, sets)
    else:
        raise _type_not_supported


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
    if issparse(F):
        return sparse.tpt.total_flux(F, A=A)
    elif isdense(F):
        return dense.tpt.total_flux(F, A=A)
    else:
        raise _type_not_supported


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
    return dense.tpt.rate(totflux, pi, qminus)


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
    return dense.tpt.mfpt(totflux, pi, qminus)


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
    if issparse(F):
        return sparse.pathways.pathways(F, A, B, fraction=fraction, maxiter=maxiter)
    elif isdense(F):
        return sparse.pathways.pathways(csr_matrix(F), A, B, fraction=fraction, maxiter=maxiter)
    else:
        raise _type_not_supported
