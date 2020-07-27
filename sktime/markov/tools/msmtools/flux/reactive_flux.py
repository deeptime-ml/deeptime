
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
from . import api as tptapi

__all__ = ['ReactiveFlux']


class ReactiveFlux(object):
    r"""A->B reactive flux from transition path theory (TPT)

    This object describes a reactive flux, i.e. a network of fluxes from a set of source states A, to a set of
    sink states B, via a set of intermediate nodes. Every node has three properties: the stationary probability mu,
    the forward committor qplus and the backward committor qminus. Every pair of edges has the following properties:
    a flux, generally a net flux that has no unnecessary back-fluxes, and optionally a gross flux.

    Flux objects can be used to compute transition pathways (and their weights) from A to B, the total flux, the
    total transition rate or mean first passage time, and they can be coarse-grained onto a set discretization
    of the node set.

    Fluxes can be computed in EMMA using transition path theory - see :func:`msmtools.tpt`

    Parameters
    ----------
    A : array_like
        List of integer state labels for set A
    B : array_like
        List of integer state labels for set B
    flux : (n,n) ndarray or scipy sparse matrix
        effective or net flux of A->B pathways
    mu : (n,) ndarray (optional)
        Stationary vector
    qminus : (n,) ndarray (optional)
        Backward committor for A->B reaction
    qplus : (n,) ndarray (optional)
        Forward committor for A-> B reaction
    gross_flux : (n,n) ndarray or scipy sparse matrix
        gross flux of A->B pathways, if available

    Notes
    -----
    Reactive flux contains a flux network from educt states (A) to product states (B).

    See also
    --------
    msmtools.tpt

    """

    def __init__(self, A, B, flux,
                 mu=None, qminus=None, qplus=None, gross_flux=None):
        # set data
        self._A = A
        self._B = B
        self._flux = flux
        self._mu = mu
        self._qminus = qminus
        self._qplus = qplus
        self._gross_flux = gross_flux
        # compute derived quantities:
        self._totalflux = tptapi.total_flux(flux, A)
        self._kAB = tptapi.rate(self._totalflux, mu, qminus)

    @property
    def nstates(self):
        r"""Returns the number of states.

        """
        return np.shape(self._flux)[0]

    @property
    def A(self):
        r"""Returns the set of reactant (source) states.

        """
        return self._A

    @property
    def B(self):
        r"""Returns the set of product (target) states

        """
        return self._B

    @property
    def I(self):
        r"""Returns the set of intermediate states

        """
        return list(set(range(self.nstates)) - set(self._A) - set(self._B))

    @property
    def stationary_distribution(self):
        r"""Returns the stationary distribution
        """
        return self._mu

    @property
    def flux(self):
        r"""Returns the effective or net flux
        """
        return self._flux

    @property
    def net_flux(self):
        r"""Returns the effective or net flux
        """
        return self._flux

    @property
    def gross_flux(self):
        r"""Returns the gross A-->B flux
        """
        return self._gross_flux

    @property
    def committor(self):
        r"""Returns the forward committor probability
        """
        return self._qplus

    @property
    def forward_committor(self):
        r"""Returns the forward committor probability
        """
        return self._qplus

    @property
    def backward_committor(self):
        r"""Returns the backward committor probability
        """
        return self._qminus

    @property
    def total_flux(self):
        r"""Returns the total flux
        """
        return self._totalflux

    @property
    def rate(self):
        r"""Returns the rate (inverse mfpt) of A-->B transitions
        """
        return self._kAB

    @property
    def mfpt(self):
        r"""Returns the rate (inverse mfpt) of A-->B transitions
        """
        return 1.0 / self._kAB

    def pathways(self, fraction=1.0, maxiter=1000):
        r"""Decompose flux network into dominant reaction paths.

        Parameters
        ----------
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

        References
        ----------
        .. [1] P. Metzner, C. Schuette and E. Vanden-Eijnden.
            Transition Path Theory for Markov Jump Processes.
            Multiscale Model Simul 7: 1192-1219 (2009)

        """
        return tptapi.pathways(self.net_flux, self.A, self.B,
                               fraction=fraction, maxiter=maxiter)

    def _pathways_to_flux(self, paths, pathfluxes, n=None):
        r"""Sums up the flux from the pathways given

        Parameters
        -----------
        paths : list of int-arrays
        list of pathways

        pathfluxes : double-array
            array with path fluxes

        n : int
            number of states. If not set, will be automatically determined.

        Returns
        -------
        flux : (n,n) ndarray of float
            the flux containing the summed path fluxes

        """
        if (n is None):
            n = 0
            for p in paths:
                n = max(n, np.max(p))
            n += 1

        # initialize flux
        F = np.zeros((n, n))
        for i in range(len(paths)):
            p = paths[i]
            for t in range(len(p) - 1):
                F[p[t], p[t + 1]] += pathfluxes[i]
        return F

    def major_flux(self, fraction=0.9):
        r"""Returns the main pathway part of the net flux comprising
        at most the requested fraction of the full flux.

        """
        (paths, pathfluxes) = self.pathways(fraction=fraction)
        return self._pathways_to_flux(paths, pathfluxes, n=self.nstates)

    # this will be a private function in tpt. only Parameter left will be the sets to be distinguished
    def _compute_coarse_sets(self, user_sets):
        r"""Computes the sets to coarse-grain the tpt flux to.

        Parameters
        ----------
            (tpt_sets, A, B) with
                tpt_sets : list of int-iterables
                sets of states that shall be distinguished in the coarse-grained flux.
            A : int-iterable
                set indexes in A
            B : int-iterable
                set indexes in B

        Returns
        -------
        sets : list of int-iterables
            sets to compute tpt on. These sets still respect the boundary between
            A, B and the intermediate tpt states.

        Notes
        -----
        Given the sets that the user wants to distinguish, the
        algorithm will create additional sets if necessary

           * If states are missing in user_sets, they will be put into a
            separate set
           * If sets in user_sets are crossing the boundary between A, B and the
             intermediates, they will be split at these boundaries. Thus each
             set in user_sets can remain intact or be split into two or three
             subsets

        """
        # set-ify everything
        setA = set(self.A)
        setB = set(self.B)
        setI = set(self.I)
        raw_sets = [set(user_set) for user_set in user_sets]

        # anything missing? Compute all listed states
        set_all = set(range(self.nstates))
        set_all_user = []
        for user_set in raw_sets:
            set_all_user += user_set
        set_all_user = set(set_all_user)
        # ... and add all the unlisted states in a separate set
        set_rest = set_all - set_all_user
        if len(set_rest) > 0:
            raw_sets.append(set_rest)

        # split sets
        Asets = []
        Isets = []
        Bsets = []
        for raw_set in raw_sets:
            s = raw_set.intersection(setA)
            if len(s) > 0:
                Asets.append(s)
            s = raw_set.intersection(setI)
            if len(s) > 0:
                Isets.append(s)
            s = raw_set.intersection(setB)
            if len(s) > 0:
                Bsets.append(s)
        tpt_sets = Asets + Isets + Bsets
        Aindexes = list(range(0, len(Asets)))
        Bindexes = list(range(len(Asets) + len(Isets), len(tpt_sets)))

        return (tpt_sets, Aindexes, Bindexes)

    def coarse_grain(self, user_sets):
        r"""Coarse-grains the flux onto user-defined sets.

        Parameters
        ----------
        user_sets : list of int-iterables
            sets of states that shall be distinguished in the coarse-grained flux.

        Returns
        -------
        (sets, tpt) : (list of int-iterables, tpt-object)
            sets contains the sets tpt is computed on. The tpt states of the new
            tpt object correspond to these sets of states in this order. Sets might
            be identical, if the user has already provided a complete partition that
            respects the boundary between A, B and the intermediates. If not, Sets
            will have more members than provided by the user, containing the
            "remainder" states and reflecting the splitting at the A and B
            boundaries.
            tpt contains a new tpt object for the coarse-grained flux. All its
            quantities (gross_flux, net_flux, A, B, committor, backward_committor)
            are coarse-grained to sets.

        Notes
        -----
        All user-specified sets will be split (if necessary) to
        preserve the boundary between A, B and the intermediate
        states.

        """
        # coarse-grain sets
        (tpt_sets, Aindexes, Bindexes) = self._compute_coarse_sets(user_sets)
        nnew = len(tpt_sets)

        # coarse-grain fluxHere we should branch between sparse and dense implementations, but currently there is only a
        F_coarse = tptapi.coarsegrain(self._gross_flux, tpt_sets)
        Fnet_coarse = tptapi.to_netflux(F_coarse)

        # coarse-grain stationary probability and committors - this can be done all dense
        pstat_coarse = np.zeros((nnew))
        forward_committor_coarse = np.zeros((nnew))
        backward_committor_coarse = np.zeros((nnew))
        for i in range(0, nnew):
            I = list(tpt_sets[i])
            muI = self._mu[I]
            pstat_coarse[i] = np.sum(muI)
            partialI = muI / pstat_coarse[i]  # normalized stationary probability over I
            forward_committor_coarse[i] = np.dot(partialI, self._qplus[I])
            backward_committor_coarse[i] = np.dot(partialI, self._qminus[I])

        res = ReactiveFlux(Aindexes, Bindexes, Fnet_coarse, mu=pstat_coarse,
                           qminus=backward_committor_coarse, qplus=forward_committor_coarse, gross_flux=F_coarse)
        return (tpt_sets, res)
