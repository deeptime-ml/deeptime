"""This module contains function for the Transition Path Theory (TPT)
analysis of Markov models.

__moduleauthor__ = "Benjamin Trendelkamp-Schroer, Frank Noe, Martin Scherer"

"""
from typing import Optional, Iterable

import numpy as np

from .tools import flux as tptapi

from deeptime.base import Model
from deeptime.markov.tools.flux import to_netflux, pathways, coarsegrain

from ..util.types import ensure_array


class ReactiveFlux(Model):
    r""" The A->B reactive flux from transition path theory (TPT).

    This object describes a reactive flux, i.e. a network of fluxes from a set of source states A, to a set of
    sink states B, via a set of intermediate nodes. Every node has three properties: the stationary probability mu,
    the forward committor qplus and the backward committor qminus. Every pair of edges has the following properties:
    a flux, generally a net flux that has no unnecessary back-fluxes, and optionally a gross flux.

    Flux objects can be used to compute transition pathways (and their weights) from A to B, the total flux, the
    total transition rate or mean first passage time, and they can be coarse-grained onto a set discretization
    of the node set.

    Fluxes can be computed using transition path theory - see :footcite:`metzner2009transition`
    and :func:`deeptime.markov.tools.tpt`.

    Parameters
    ----------
    source_states : array_like
        List of integer state labels for set A
    target_states : array_like
        List of integer state labels for set B
    net_flux : (n,n) ndarray or scipy sparse matrix
        effective or net flux of A->B pathways
    stationary_distribution : (n,) ndarray (optional)
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
    reactive_flux : Method that produces ReactiveFlux instances
    deeptime.markov.msm.MarkovStateModel.reactive_flux : TPT analysis based on a Markov state model

    References
    ----------
    .. footbibliography::
    """

    def __init__(self, source_states, target_states, net_flux, stationary_distribution=None,
                 qminus=None, qplus=None, gross_flux=None):
        # set data
        super().__init__()
        self._source_states = source_states
        self._target_states = target_states
        self._net_flux = net_flux
        self._stationary_distribution = stationary_distribution
        self._qminus = qminus
        self._qplus = qplus
        self._gross_flux = gross_flux
        # compute derived quantities:
        self._totalflux = tptapi.total_flux(net_flux, source_states)
        self._kAB = tptapi.rate(self._totalflux, stationary_distribution, qminus)

    @property
    def n_states(self):
        """number of states."""
        return self._net_flux.shape[0]

    @property
    def source_states(self):
        """set of reactant (source) states."""
        return self._source_states

    @property
    def target_states(self):
        """set of product (target) states"""
        return self._target_states

    @property
    def intermediate_states(self):
        """set of intermediate states"""
        return list(set(range(self.n_states)) - set(self._source_states) - set(self._target_states))

    @property
    def stationary_distribution(self):
        """stationary distribution"""
        return self._stationary_distribution

    @property
    def net_flux(self):
        r""" Effective or net flux. Units are :math:`1/ \mathrm{time}`. """
        return self._net_flux

    @property
    def gross_flux(self):
        r""" Gross :math:`A\\rightarrow B` flux. Units are :math:`1/ \mathrm{time}`. """
        return self._gross_flux

    @property
    def forward_committor(self):
        """forward committor probability"""
        return self._qplus

    @property
    def backward_committor(self):
        """backward committor probability"""
        return self._qminus

    @property
    def total_flux(self):
        r""" The total flux. Units are :math:`1/ \mathrm{time}`. """
        return self._totalflux

    @property
    def rate(self):
        r""" Rate (inverse mfpt) of :math:`A\\rightarrow B` transitions in units of :math:`1/ \mathrm{time}`. """
        return self._kAB

    @property
    def mfpt(self):
        """ Mean-first-passage-time (inverse rate) of :math:`A\\rightarrow B` transitions. """
        return 1. / self._kAB

    def pathways(self, fraction=1.0, maxiter=1000):
        """Decompose flux network into dominant reaction paths.

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
        """
        return pathways(self._net_flux, self.source_states, self.target_states,
                        fraction=fraction, maxiter=maxiter)

    @staticmethod
    def _pathways_to_flux(paths, pathfluxes, n=None):
        """Sums up the flux from the pathways given

        Parameters
        -----------
        paths : list of int-arrays
        list of pathways

        pathfluxes : double-array
            array with path fluxes

        n : int or None
            number of states. If not set, will be automatically determined.

        Returns
        -------
        flux : (n,n) ndarray of float
            the flux containing the summed path fluxes

        """
        if n is None:
            n = 0
            for p in paths:
                n = max(n, np.max(p))
            n += 1

        # initialize flux
        flux = np.zeros((n, n))
        for i, p in enumerate(paths):
            for t in range(len(p) - 1):
                flux[p[t], p[t + 1]] += pathfluxes[i]
        return flux

    def major_flux(self, fraction=0.9):
        """ Returns the main pathway part of the net flux comprising at most the requested fraction of the full flux.
        """
        paths, pathfluxes = self.pathways(fraction=fraction)
        return self._pathways_to_flux(paths, pathfluxes, n=self.n_states)

    # this will be a private function in tpt. only Parameter left will be the sets to be distinguished
    def _compute_coarse_sets(self, user_sets):
        """Computes the sets to coarse-grain the tpt flux to.

        Parameters
        ----------
        user_sets : list of int-iterables
            sets of states that shall be distinguished in the coarse-grained flux.

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
        setA = set(self.source_states)
        setB = set(self.target_states)
        setI = set(self.intermediate_states)
        raw_sets = [set(user_set) for user_set in user_sets]

        # anything missing? Compute all listed states
        set_all = set(range(self.n_states))
        set_all_user = []
        for user_set in raw_sets:
            set_all_user += user_set
        set_all_user = set(set_all_user)
        # ... and add all the unlisted states in a separate set
        set_rest = set_all - set_all_user
        if len(set_rest) > 0:
            raw_sets.append(set_rest)

        # split sets
        source_sets = []
        intermediate_sets = []
        target_sets = []
        for raw_set in raw_sets:
            s = raw_set.intersection(setA)
            if len(s) > 0:
                source_sets.append(s)
            s = raw_set.intersection(setI)
            if len(s) > 0:
                intermediate_sets.append(s)
            s = raw_set.intersection(setB)
            if len(s) > 0:
                target_sets.append(s)
        tpt_sets = source_sets + intermediate_sets + target_sets
        source_indexes = list(range(0, len(source_sets)))
        target_indexes = list(range(len(source_sets) + len(intermediate_sets), len(tpt_sets)))

        return tpt_sets, source_indexes, target_indexes

    def coarse_grain(self, user_sets):
        """Coarse-grains the flux onto user-defined sets.

        Parameters
        ----------
        user_sets : list of int-iterables
            sets of states that shall be distinguished in the coarse-grained flux.

        Returns
        -------
        (sets, tpt) : (list of int-iterables, ReactiveFlux)
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
        tpt_sets, source_indices, target_indices = self._compute_coarse_sets(user_sets)
        nnew = len(tpt_sets)

        # coarse-grain flux
        # Here we should branch between sparse and dense implementations, but currently there is only a dense version.
        flux_coarse = coarsegrain(self._gross_flux, tpt_sets)
        net_flux_coarse = to_netflux(flux_coarse)

        # coarse-grain stationary probability and committors - this can be done all dense
        pstat_coarse = np.zeros(nnew)
        forward_committor_coarse = np.zeros(nnew)
        backward_committor_coarse = np.zeros(nnew)
        for i in range(0, nnew):
            I = list(tpt_sets[i])
            statdist_subselection = self._stationary_distribution[I]
            pstat_coarse[i] = np.sum(statdist_subselection)
            # normalized stationary probability over I
            statdist_subselection_prob = statdist_subselection / pstat_coarse[i]
            forward_committor_coarse[i] = np.dot(statdist_subselection_prob, self._qplus[I])
            backward_committor_coarse[i] = np.dot(statdist_subselection_prob, self._qminus[I])

        res = ReactiveFlux(source_indices, target_indices, net_flux=net_flux_coarse,
                           stationary_distribution=pstat_coarse, qminus=backward_committor_coarse,
                           qplus=forward_committor_coarse, gross_flux=flux_coarse)
        return tpt_sets, res


def reactive_flux(transition_matrix: np.ndarray, source_states: Iterable[int], target_states: Iterable[int],
                  stationary_distribution=None, qminus=None, qplus=None,
                  transition_matrix_tolerance: Optional[float] = None) -> ReactiveFlux:
    r""" Computes the A->B reactive flux using transition path theory (TPT).

    Parameters
    ----------
    transition_matrix : (M, M) ndarray or scipy.sparse matrix
        The transition matrix.
    source_states : array_like
        List of integer state labels for set A
    target_states : array_like
        List of integer state labels for set B
    stationary_distribution : (M,) ndarray, optional, default=None
        Stationary vector. If None is computed from the transition matrix internally.
    qminus : (M,) ndarray (optional)
        Backward committor for A->B reaction
    qplus : (M,) ndarray (optional)
        Forward committor for A-> B reaction
    transition_matrix_tolerance : float, optional, default=None
        Tolerance with which is checked whether the input is actually a transition matrix. If None (default),
        no check is performed.

    Returns
    -------
    tpt: deeptime.markov.tools.flux.ReactiveFlux object
        A python object containing the reactive A->B flux network
        and several additional quantities, such as stationary probability,
        committors and set definitions.

    Notes
    -----
    The central object used in transition path theory is the forward and backward comittor function.

    TPT (originally introduced in :footcite:`weinan2006towards`) for continous systems has a
    discrete version outlined in :footcite:`metzner2009transition`. Here, we use the transition
    matrix formulation described in :footcite:`noe2009constructing`.

    See also
    --------
    ReactiveFlux

    References
    ----------
    .. footbibliography::
    """
    import deeptime.markov.tools.analysis as msmana

    source_states = ensure_array(source_states, dtype=int)
    target_states = ensure_array(target_states, dtype=int)

    if len(source_states) == 0 or len(target_states) == 0:
        raise ValueError('set A or B is empty')

    n_states = transition_matrix.shape[0]
    if len(source_states) > n_states or len(target_states) > n_states \
            or max(source_states) > n_states or max(target_states) > n_states:
        raise ValueError('set A or B defines more states than the given transition matrix.')

    if transition_matrix_tolerance is not None and \
            msmana.is_transition_matrix(transition_matrix, tol=transition_matrix_tolerance):
        raise ValueError('given matrix T is not a transition matrix')

    # we can compute the following properties from either dense or sparse T
    # stationary dist
    if stationary_distribution is None:
        stationary_distribution = msmana.stationary_distribution(transition_matrix)
    # forward committor
    if qplus is None:
        qplus = msmana.committor(transition_matrix, source_states, target_states, forward=True)
    # backward committor
    if qminus is None:
        if msmana.is_reversible(transition_matrix, mu=stationary_distribution):
            qminus = 1.0 - qplus
        else:
            qminus = msmana.committor(transition_matrix, source_states, target_states, forward=False,
                                      mu=stationary_distribution)
    # gross flux
    grossflux = tptapi.flux_matrix(transition_matrix, stationary_distribution, qminus, qplus, netflux=False)
    # net flux
    netflux = to_netflux(grossflux)

    # construct flux object
    return ReactiveFlux(source_states, target_states, net_flux=netflux,
                        stationary_distribution=stationary_distribution,
                        qminus=qminus, qplus=qplus, gross_flux=grossflux)
