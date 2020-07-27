
# This file is part of MSMTools.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# MSMTools is free software: you can redistribute it and/or modify
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

r"""This module contains function for the Transition Path Theory (TPT)
analysis of Markov models.

__moduleauthor__ = "Benjamin Trendelkamp-Schroer, Frank Noe"

"""
import numpy as np

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
    (TPT). The central object used in transition path theory is the
    forward and backward comittor function.

    See also
    --------
    committor.forward_committor, committor.backward_committor


    """
    ind = np.diag_indices(T.shape[0])
    flux = pi[:, np.newaxis] * qminus[:, np.newaxis] * T * qplus[np.newaxis, :]
    """Remove self fluxes f_ii"""
    flux[ind] = 0.0
    """Return net or gross flux"""
    if netflux:
        return to_netflux(flux)
    else:
        return flux


def to_netflux(flux):
    r"""Compute the netflux from the gross flux.

        f_ij^{+}=max{0, f_ij-f_ji}
        for all pairs i,j

    Parameters
    ----------
    flux : (M, M) ndarray
        Matrix of flux values between pairs of states.

    Returns
    -------
    netflux : (M, M) ndarray
        Matrix of netflux values between pairs of states.

    """
    netflux = flux - np.transpose(flux)
    """Set negative fluxes to zero"""
    ind = (netflux < 0.0)
    netflux[ind] = 0.0
    return netflux


def flux_production(F):
    r"""Returns the net flux production for all states

    Parameters
    ----------
    F : (n, n) ndarray
        Matrix of flux values between pairs of states.

    Returns
    -------
    prod : (n) ndarray
        array with flux production (positive) or consumption (negative) at each state
    """
    influxes = np.array(np.sum(F, axis=0)).flatten()  # all that flows in
    outfluxes = np.array(np.sum(F, axis=1)).flatten()  # all that flows out
    prod = outfluxes - influxes  # net flux into nodes
    return prod


def flux_producers(F, rtol=1e-05, atol=1e-12):
    r"""Return indexes of states that are net flux producers.

    Parameters
    ----------
    F : (n, n) ndarray
        Matrix of flux values between pairs of states.
    rtol : float
        relative tolerance. fulfilled if max(outflux-influx, 0) / max(outflux,influx) < rtol
    atol : float
        absolute tolerance. fulfilled if max(outflux-influx, 0) < atol

    Returns
    -------
    producers : (n) ndarray of int
        indexes of states that are net flux producers. May include "dirty" producers, i.e.
        states that have influx but still produce more outflux and thereby violate flux
        conservation.

    """
    n = F.shape[0]
    influxes = np.array(np.sum(F, axis=0)).flatten()  # all that flows in
    outfluxes = np.array(np.sum(F, axis=1)).flatten()  # all that flows out
    # net out flux absolute
    prod_abs = np.maximum(outfluxes - influxes, np.zeros(n))
    # net out flux relative
    prod_rel = prod_abs / (np.maximum(outfluxes, influxes))
    # return all indexes that are produces in terms of absolute and relative tolerance
    return list(np.where((prod_abs > atol) * (prod_rel > rtol))[0])


def flux_consumers(F, rtol=1e-05, atol=1e-12):
    r"""Return indexes of states that are net flux producers.

    Parameters
    ----------
    F : (n, n) ndarray
        Matrix of flux values between pairs of states.
    rtol : float
        relative tolerance. fulfilled if max(outflux-influx, 0) / max(outflux,influx) < rtol
    atol : float
        absolute tolerance. fulfilled if max(outflux-influx, 0) < atol

    Returns
    -------
    producers : (n) ndarray of int
        indexes of states that are net flux producers. May include "dirty" producers, i.e.
        states that have influx but still produce more outflux and thereby violate flux
        conservation.

    """
    # can be used with sparse or dense
    n = np.shape(F)[0]
    influxes = np.array(np.sum(F, axis=0)).flatten()  # all that flows in
    outfluxes = np.array(np.sum(F, axis=1)).flatten()  # all that flows out
    # net in flux absolute
    con_abs = np.maximum(influxes - outfluxes, np.zeros(n))
    # net in flux relative
    con_rel = con_abs / (np.maximum(outfluxes, influxes))
    # return all indexes that are produces in terms of absolute and relative tolerance
    return list(np.where((con_abs > atol) * (con_rel > rtol))[0])


def coarsegrain(F, sets):
    r"""Coarse-grains the flux to the given sets

    $fc_{i,j} = \sum_{i \in I,j \in J} f_{i,j}$
    Note that if you coarse-grain a net flux, it does not necessarily have a net
    flux property anymore. If want to make sure you get a netflux,
    use to_netflux(coarsegrain(F,sets)).

    Parameters
    ----------
    F : (n, n) ndarray
        Matrix of flux values between pairs of states.
    sets : list of array-like of ints
        The sets of states onto which the flux is coarse-grained.

    """
    nnew = len(sets)
    Fc = np.zeros((nnew, nnew))
    for i in range(0, nnew - 1):
        for j in range(i + 1, nnew):
            I = list(sets[i])
            J = list(sets[j])
            Fc[i, j] = np.sum(F[I, :][:, J])
            Fc[j, i] = np.sum(F[J, :][:, I])
    return Fc


# ======================================================================
# Total flux, rate and mfpt for the A->B reaction
# ======================================================================


def total_flux(F, A=None):
    r"""Compute the total flux, or turnover flux, that is produced by the
        flux sources and consumed by the flux sinks

    Parameters
    ----------
    F : (n, n) ndarray
        Matrix of flux values between pairs of states.
    A : array_like (optional)
        List of integer state labels for set A (reactant)

    Returns
    -------
    F : float
        The total flux, or turnover flux, that is produced by the
        flux sources and consumed by the flux sinks

    """
    if A is None:
        prod = flux_production(F)
        zeros = np.zeros(len(prod))
        outflux = np.sum(np.maximum(prod, zeros))
        return outflux
    else:
        X = set(np.arange(F.shape[0]))  # total state space
        A = set(A)
        notA = X.difference(A)
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
    kAB : float
        The reaction rate (per time step of the
        Markov chain)

    """
    return 1.0 / rate(totflux, pi, qminus)
