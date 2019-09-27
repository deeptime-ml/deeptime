"""This module contains function for the Transition Path Theory (TPT)
analysis of Markov models.

__moduleauthor__ = "Benjamin Trendelkamp-Schroer, Frank Noe, Martin Scherer"

"""
import numpy as np
from msmtools.analysis import committor
from msmtools.flux import (to_netflux, flux_matrix, total_flux, rate, pathways, coarsegrain)

from sktime.base import Model, Estimator
from sktime.markovprocess import Q_

__all__ = ['ReactiveFlux']


class ReactiveFlux(Model):
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
    flux : (n, n) ndarray or scipy sparse matrix
        effective or net flux of A->B pathways
    stationary_distribution : (n,) ndarray (optional)
        Stationary vector
    qminus : (n,) ndarray (optional)
        Backward committor for A->B reaction
    qplus : (n,) ndarray (optional)
        Forward committor for A-> B reaction
    gross_flux : (n,n) ndarray or scipy sparse matrix
        gross flux of A->B pathways, if available
    dt_model : Quantity or None, optional
        when the originating model has a lag time, output units will be scaled by it.

    Notes
    -----
    Reactive flux contains a flux network from educt states (A) to product states (B).

    See also
    --------
    msmtools.tpt

    """

    def __init__(self, A=None, B=None, flux=None,
                 stationary_distribution=None, qminus=None, qplus=None, gross_flux=None, dt_model=1):
        # set data
        self.A = A
        self.B = B
        self._flux = flux
        self._mu = stationary_distribution
        self._qminus = qminus
        self._qplus = qplus
        self._gross_flux = gross_flux
        self.dt_model = dt_model
        if flux is not None and A is not None and stationary_distribution is not None and qminus is not None:
            # compute derived quantities:
            self._totalflux = total_flux(flux, A)
            self._kAB = rate(self._totalflux, stationary_distribution, qminus)

    @property
    def dt_model(self) -> Q_:
        return self._dt_model

    @dt_model.setter
    def dt_model(self, value):
        self._dt_model = Q_(value)

    @property
    def nstates(self):
        """number of states."""
        return self._flux.shape[0]

    @property
    def A(self):
        """set of reactant (source) states."""
        return self._A

    @A.setter
    def A(self, val):
        self._A = val

    @property
    def B(self):
        """set of product (target) states"""
        return self._B

    @B.setter
    def B(self, val):
        self._B = val

    @property
    def I(self):
        """set of intermediate states"""
        return list(set(range(self.nstates)) - set(self._A) - set(self._B))

    @property
    def stationary_distribution(self):
        """stationary distribution"""
        return self._mu

    @stationary_distribution.setter
    def stationary_distribution(self, val):
        self._mu = val

    @property
    def net_flux(self):
        """effective or net flux"""
        return self._flux / self._dt_model

    @property
    def gross_flux(self):
        """gross A-->B flux"""
        return self._gross_flux / self._dt_model

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
        """total flux"""
        return self._totalflux / self._dt_model

    @property
    def rate(self):
        """rate (inverse mfpt) of A-->B transitions"""
        return self._kAB / self._dt_model

    @property
    def mfpt(self):
        """mean-first-passage-time (inverse rate) of A-->B transitions"""
        return self._dt_model / self._kAB

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

        References
        ----------
        .. [1] P. Metzner, C. Schuette and E. Vanden-Eijnden.
            Transition Path Theory for Markov Jump Processes.
            Multiscale Model Simul 7: 1192-1219 (2009)

        """
        return pathways(self._flux, self.A, self.B,
                        fraction=fraction, maxiter=maxiter)

    def _pathways_to_flux(self, paths, pathfluxes, n=None):
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
        F = np.zeros((n, n))
        for i, p in enumerate(paths):
            for t in range(len(p) - 1):
                F[p[t], p[t + 1]] += pathfluxes[i]
        return F

    def major_flux(self, fraction=0.9):
        """ main pathway part of the net flux comprising at most the requested fraction of the full flux.
        """
        paths, pathfluxes = self.pathways(fraction=fraction)
        return self._pathways_to_flux(paths, pathfluxes, n=self.nstates)

    # this will be a private function in tpt. only Parameter left will be the sets to be distinguished
    def _compute_coarse_sets(self, user_sets):
        """Computes the sets to coarse-grain the tpt flux to.

        Parameters
        ----------
        tpt_sets : list of int-iterables
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

        return tpt_sets, Aindexes, Bindexes

    def coarse_grain(self, user_sets):
        """Coarse-grains the flux onto user-defined sets.

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
        tpt_sets, Aindexes, Bindexes = self._compute_coarse_sets(user_sets)
        nnew = len(tpt_sets)

        # coarse-grain fluxHere we should branch between sparse and dense implementations, but currently there is only a
        F_coarse = coarsegrain(self._gross_flux, tpt_sets)
        Fnet_coarse = to_netflux(F_coarse)

        # coarse-grain stationary probability and committors - this can be done all dense
        pstat_coarse = np.zeros(nnew)
        forward_committor_coarse = np.zeros(nnew)
        backward_committor_coarse = np.zeros(nnew)
        for i in range(0, nnew):
            I = list(tpt_sets[i])
            muI = self._mu[I]
            pstat_coarse[i] = np.sum(muI)
            partialI = muI / pstat_coarse[i]  # normalized stationary probability over I
            forward_committor_coarse[i] = np.dot(partialI, self._qplus[I])
            backward_committor_coarse[i] = np.dot(partialI, self._qminus[I])

        res = ReactiveFlux(Aindexes, Bindexes, Fnet_coarse, stationary_distribution=pstat_coarse,
                           qminus=backward_committor_coarse, qplus=forward_committor_coarse, gross_flux=F_coarse,
                           dt_model=self.dt_model)
        return tpt_sets, res


class ReactiveFluxEstimator(Estimator):
    r""" A->B reactive flux from transition path theory (TPT)

       The estimated model :class:`ReactiveFlux`
       can be used to extract various quantities of the flux, as well as to
       compute A -> B transition pathways, their weights, and to coarse-grain
       the flux onto sets of states.

       Parameters
       ----------
       A : array_like
           List of integer state labels for set A
       B : array_like
           List of integer state labels for set B

       Returns
       -------
       tptobj : :class:`ReactiveFlux <pyemma.msm.models.ReactiveFlux>` object
           An object containing the reactive A->B flux network
           and several additional quantities, such as the stationary probability,
           committors and set definitions.

       See also
       --------
       :class:`ReactiveFlux`
           Reactive Flux model

       References
       ----------
       Transition path theory was introduced for space-continuous dynamical
       processes, such as Langevin dynamics, in [1]_, [2]_ introduces discrete
       transition path theory for Markov jump processes (Master equation models,
       rate matrices) and pathway decomposition algorithms. [3]_ introduces
       transition path theory for Markov state models (MSMs) and some analysis
       algorithms. In this function, the equations described in [3]_ are applied.

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


       Computes the A->B reactive flux using transition path theory (TPT)

       Parameters
       ----------
       T : (M, M) ndarray or scipy.sparse matrix
           Transition matrix (default) or Rate matrix (if rate_matrix=True)
       A : array_like
           List of integer state labels for set A
       B : array_like
           List of integer state labels for set B
       mu : (M,) ndarray (optional)
           Stationary vector
       qminus : (M,) ndarray (optional)
           Backward committor for A->B reaction
       qplus : (M,) ndarray (optional)
           Forward committor for A-> B reaction
       rate_matrix = False : boolean
           By default (False), T is a transition matrix.
           If set to True, T is a rate matrix.

       Notes
       -----
       The central object used in transition path theory is
       the forward and backward comittor function.

       TPT (originally introduced in [1]) for continuous systems has a
       discrete version outlined in [2]. Here, we use the transition
       matrix formulation described in [3].

       See also
       --------
       msmtools.analysis.committor, ReactiveFlux

       References
       ----------
       .. [1] W. E and E. Vanden-Eijnden.
           Towards a theory of transition paths.
           J. Stat. Phys. 123: 503-523 (2006)
       .. [2] P. Metzner, C. Schuette and E. Vanden-Eijnden.
           Transition Path Theory for Markov Jump Processes.
           Multiscale Model Simul 7: 1192-1219 (2009)
       .. [3] F. Noe, Ch. Schuette, E. Vanden-Eijnden, L. Reich and T. Weikl:
           Constructing the Full Ensemble of Folding Pathways from Short Off-Equilibrium Simulations.
           Proc. Natl. Acad. Sci. USA, 106, 19011-19016 (2009)

       """

    def __init__(self, A, B):
        if len(A) == 0:
            raise ValueError('set A is empty')
        if len(B) == 0:
            raise ValueError('set B is empty')
        self.A = A
        self.B = B
        super(ReactiveFluxEstimator, self).__init__()

    def fit(self, msm):
        """

        :param msm:
        :return:
        """
        T = msm.transition_matrix
        mu = msm.stationary_distribution
        n = T.shape[0]
        if len(self.A) > n or max(self.A) > n:
            raise ValueError(f'set A defines more states ({self.A}), than given transition matrix (nstates={n}).')
        if len(self.B) > n or max(self.B) > n:
            raise ValueError(f'set B defines more states ({self.B}), than given transition matrix (nstates={n}).')

        # forward committor
        qplus = committor(T, self.A, self.B, forward=True)
        # backward committor
        if msm.is_reversible:
            qminus = 1.0 - qplus
        else:
            qminus = committor(T, self.A, self.B, forward=False, mu=mu)
        # gross flux
        grossflux = flux_matrix(T, mu, qminus, qplus, netflux=False)
        # net flux
        netflux = to_netflux(grossflux)

        self._model.__init__(self.A, self.B, netflux, stationary_distribution=mu, qminus=qminus, qplus=qplus,
                             gross_flux=grossflux, dt_model=msm.dt_model)
        return self

    def _create_model(self):
        return ReactiveFlux()
