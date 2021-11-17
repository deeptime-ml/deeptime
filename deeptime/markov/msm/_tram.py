import logging
from typing import Optional

from ._markov_state_model import MarkovStateModelCollection
from deeptime.markov import TransitionCountEstimator, count_states
from .._base import _MSMBaseEstimator
from deeptime.util import types
from deeptime.markov._tram_bindings import tram
from deeptime.markov import _markov_bindings, compute_connected_sets, cset

import numpy as np

__all__ = ['TRAM']

log = logging.getLogger(__file__)


class TRAM(_MSMBaseEstimator):
    r"""
    Parameters
    ----------
    References
    ----------
    """

    def __init__(
            self, lagtime=None, count_mode='sliding',
            connectivity='post_hoc_RE',
            nstates_full=None, equilibrium=None,
            maxiter=10000, maxerr: float = 1.0E-15, save_convergence_info=0,
            nn=None, connectivity_factor: float = 1.0, connectivity_threshold: float = 0,
            direct_space=False, N_dtram_accelerations=0,
            callback=None,
            init='mbar', init_maxiter=5000, init_maxerr: float = 1.0E-8,
            overcounting_factor=1.0):
        r"""Transition(-based) Reweighting Analysis Method
        Parameters
        ----------
        lag : int
            Integer lag time at which transitions are counted.
        count_mode : str, optional, default='sliding'
            mode to obtain count matrices from discrete trajectories. Should be
            one of:
            * 'sliding' : A trajectory of length T will have :math:`T-\tau` counts at time indexes
                  .. math::
                     (0 \rightarrow \tau), (1 \rightarrow \tau+1), ..., (T-\tau-1 \rightarrow T-1)
            * 'sample' : A trajectory of length T will have :math:`T/\tau` counts
              at time indexes
                  .. math::
                        (0 \rightarrow \tau), (\tau \rightarrow 2 \tau), ..., ((T/\tau-1) \tau \rightarrow T)
            Currently only 'sliding' is supported.
        connectivity : str, optional, default='post_hoc_RE'
            One of 'post_hoc_RE', 'BAR_variance', 'reversible_pathways' or
            'summed_count_matrix'. Defines what should be considered a connected set
            in the joint (product) space of conformations and thermodynamic ensembles.
            * 'reversible_pathways' : requires that every state in the connected set
              can be reached by following a pathway of reversible transitions. A
              reversible transition between two Markov states (within the same
              thermodynamic state k) is a pair of Markov states that belong to the
              same strongly connected component of the count matrix (from
              thermodynamic state k). A pathway of reversible transitions is a list of
              reversible transitions [(i_1, i_2), (i_2, i_3),..., (i_(N-2), i_(N-1)),
              (i_(N-1), i_N)]. The thermodynamic state where the reversible
              transitions happen, is ignored in constructing the reversible pathways.
              This is equivalent to assuming that two ensembles overlap at some Markov
              state whenever there exist frames from both ensembles in that Markov
              state.
            * 'post_hoc_RE' : similar to 'reversible_pathways' but with a more strict
              requirement for the overlap between thermodynamic states. It is required
              that every state in the connected set can be reached by following a
              pathway of reversible transitions or jumping between overlapping
              thermodynamic states while staying in the same Markov state. A reversible
              transition between two Markov states (within the same thermodynamic
              state k) is a pair of Markov states that belong to the same strongly
              connected component of the count matrix (from thermodynamic state k).
              Two thermodynamic states k and l are defined to overlap at Markov state
              n if a replica exchange simulation [2]_ restricted to state n would show
              at least one transition from k to l or one transition from from l to k.
              The expected number of replica exchanges is estimated from the
              simulation data. The minimal number required of replica exchanges
              per Markov state can be increased by decreasing `connectivity_factor`.
            * 'BAR_variance' : like 'post_hoc_RE' but with a different condition to
              define the thermodynamic overlap based on the variance of the BAR
              estimator [3]_. Two thermodynamic states k and l are defined to overlap
              at Markov state n if the variance of the free energy difference Delta
              f_{kl} computed with BAR (and restricted to conformations form Markov
              state n) is less or equal than one. The minimally required variance
              can be controlled with `connectivity_factor`.
            * 'summed_count_matrix' : all thermodynamic states are assumed to overlap.
              The connected set is then computed by summing the count matrices over
              all thermodynamic states and taking it's largest strongly connected set.
              Not recommended!
            For more details see :func:`pyemma.thermo.extensions.cset.compute_csets_TRAM`.
        nstates_full : int, optional, default=None
            Number of cluster centers, i.e., the size of the full set of states.
        equilibrium : list of booleans, optional
            For every trajectory triple (ttraj[i], dtraj[i], btraj[i]), indicates
            whether to assume global equilibrium. If true, the triple is not used
            for computing kinetic quantities (but only thermodynamic quantities).
            By default, no trajectory is assumed to be in global equilibrium.
            This is the TRAMMBAR extension.
        maxiter : int, optional, default=10000
            The maximum number of self-consistent iterations before the estimator exits unsuccessfully.
        maxerr : float, optional, default=1E-15
            Convergence criterion based on the maximal free energy change in a self-consistent
            iteration step.
        save_convergence_info : int, optional, default=0
            Every save_convergence_info iteration steps, store the actual increment
            and the actual log-likelihood; 0 means no storage.
        connectivity_factor : float, optional, default=1.0
            Only needed if connectivity='post_hoc_RE' or 'BAR_variance'. Values
            greater than 1.0 weaken the connectivity conditions. For 'post_hoc_RE'
            this multiplies the number of hypothetically observed transitions. For
            'BAR_variance' this scales the threshold for the minimal allowed variance
            of free energy differences.
        direct_space : bool, optional, default=False
            Whether to perform the self-consistent iteration with Boltzmann factors
            (direct space) or free energies (log-space). When analyzing data from
            multi-temperature simulations, direct-space is not recommended.
        N_dtram_accelerations : int, optional, default=0
            Convergence of TRAM can be speeded up by interleaving the updates
            in the self-consistent iteration with a dTRAM-like update step.
            N_dtram_accelerations says how many times the dTRAM-like update
            step should be applied in every iteration of the TRAM equations.
            Currently this is only effective if direct_space=True.
        init : str, optional, default=None
            Use a specific initialization for self-consistent iteration:
            | None:    use a hard-coded guess for free energies and Lagrangian multipliers
            | 'mbar':  perform a short MBAR estimate to initialize the free energies
        init_maxiter : int, optional, default=5000
            The maximum number of self-consistent iterations during the initialization.
        init_maxerr : float, optional, default=1.0E-8
            Convergence criterion for the initialization.
        overcounting_factor : double, default = 1.0
            Only needed if equilibrium contains True (TRAMMBAR).
            Sets the relative statistical weight of equilibrium and non-equilibrium
            frames. An overcounting_factor of value n means that every
            non-equilibrium frame is counted n times. Values larger than 1 increase
            the relative weight of the non-equilibrium data. Values less than 1
            increase the relative weight of the equilibrium data.
        References
        ----------
        .. [1] Wu, H. et al 2016
            Multiensemble Markov models of molecular thermodynamics and kinetics
            Proc. Natl. Acad. Sci. USA 113 E3221--E3230
        .. [2]_ Hukushima et al, Exchange Monte Carlo method and application to spin
            glass simulations, J. Phys. Soc. Jan. 65, 1604 (1996)
        .. [3]_ Shirts and Chodera, Statistically optimal analysis of samples
            from multiple equilibrium states, J. Chem. Phys. 129, 124105 (2008)
        """
        super(TRAM, self).__init__()

        self.lagtime = lagtime
        assert count_mode == 'sliding', 'Currently the only implemented count_mode is \'sliding\''
        self.count_mode = count_mode
        self.connectivity = connectivity
        self.nn = nn
        self.connectivity_factor = connectivity_factor
        self.nstates_full = nstates_full
        self.equilibrium = equilibrium
        self.maxiter = maxiter
        self.maxerr = maxerr
        self.direct_space = direct_space
        self.N_dtram_accelerations = N_dtram_accelerations
        self.callback = callback
        self.save_convergence_info = save_convergence_info
        assert init in (None, 'mbar'), 'Currently only None and \'mbar\' are supported'
        self.init = init
        self.init_maxiter = init_maxiter
        self.init_maxerr = init_maxerr
        self.overcounting_factor = overcounting_factor
        self.active_set = None
        self.biased_conf_energies = None
        self.therm_energies = None
        self.log_lagrangian_mult = None
        self.loglikelihoods = None

    def fetch_model(self) -> Optional[MarkovStateModelCollection]:
        r"""Yields the most recent :class:`MarkovStateModelCollection` that was estimated.
        Can be None if fit was not called.

        Returns
        -------
        model : MarkovStateModelCollection or None
            The most recent markov state model or None.
        """
        return self._model

    def cluster_and_fit(self):
        r""" Discretize given data according to chosen clustering method and fit model to clustered data. """
        # step 1: cluster data
        # step 2: call fit_from_discrete_timeseries
        pass

    def fit_from_clustered_data(self):
        r""" Fits a model directly from given timeseries that has been discretized into Markov states by the user. """
        # step 1: make count matrices
        # step 2: call fit_from_count_matrices
        pass

    def fit_from_count_matrices(self, count_matrices, state_counts, bias_energy_sequences, state_sequences,
                                maxiter=1000, maxerr=1.0E-8, save_convergence_info=0,
                                biased_conf_energies=None, log_lagrangian_mult=None):
        r""" Fits a model directly from given timeseries that has been discretized into Markov states by the user.

        ----------
        count_matrices : numpy.ndarray(shape=(T, M, M), dtype=numpy.intc)
            transition count matrices for all T thermodynamic states
        state_counts : numpy.ndarray(shape=(T, M), dtype=numpy.intc)
            state counts for all M discrete and T thermodynamic states
        bias_energy_sequences : list of numpy.ndarray(shape=(X_i, T), dtype=numpy.float64)
            reduced bias energies in the T thermodynamic states for all X samples
        state_sequences : list of numpy.ndarray(shape=(X_i), dtype=numpy.float64)
            discrete state indices for all X samples
        maxiter : int
            maximum number of iterations
        maxerr : float
            convergence criterion based on absolute change in free energies
        save_convergence_info : int, optional
            every save_convergence_info iteration steps, store the actual increment
            and the actual loglikelihood
        biased_conf_energies : numpy.ndarray(shape=(T, M), dtype=numpy.float64), OPTIONAL
            initial guess for the reduced discrete state free energies for all T thermodynamic states
        log_lagrangian_mult : numpy.ndarray(shape=(T, M), dtype=numpy.float64), OPTIONAL
            initial guess for the logarithm of the Lagrangian multipliers

        Returns
        -------
        biased_conf_energies : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
            reduced discrete state free energies for all T thermodynamic states
        conf_energies : numpy.ndarray(shape=(M), dtype=numpy.float64)
            reduced unbiased discrete state free energies
        therm_energies : numpy.ndarray(shape=(M), dtype=numpy.float64)
            reduced thermodynamic free energies
        log_lagrangian_mult : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
            logarithm of the Lagrangian multipliers
        increments : numpy.ndarray(dtype=numpy.float64, ndim=1)
            stored sequence of increments
        loglikelihoods : numpy.ndarray(dtype=numpy.float64, ndim=1)
            stored sequence of loglikelihoods
        Note
        ----
        The self-consitent iteration terminates when
        .. math::
           \max\{\max_{i,k}{\Delta \pi_i^k}, \max_k \Delta f^k \}<\mathrm{maxerr}.
        Different termination criteria can be implemented with the callback
        function. Raising `CallbackInterrupt` in the callback will cleanly
        terminate the iteration.
        """
        if biased_conf_energies is None:
            biased_conf_energies = np.zeros(shape=state_counts.shape, dtype=np.float64)
        if log_lagrangian_mult is None:
            log_lagrangian_mult = np.zeros(shape=state_counts.shape, dtype=np.float64)
        tram.init_lagrangian_mult(count_matrices, self.nthermo, self.nstates_full, log_lagrangian_mult)
        increments = []
        loglikelihoods = []
        sci_count = 0
        assert len(state_sequences) == len(bias_energy_sequences)
        for s, b in zip(state_sequences, bias_energy_sequences):
            assert s.ndim == 1
        assert s.dtype == np.intc
        assert b.ndim == 2
        assert b.dtype == np.float64
        assert s.shape[0] == b.shape[0]
        assert b.shape[1] == count_matrices.shape[0]
        assert s.flags.c_contiguous
        assert b.flags.c_contiguous
        log_R_K_i = np.zeros(shape=state_counts.shape, dtype=np.float64)
        scratch_T = np.zeros(shape=(count_matrices.shape[0],), dtype=np.float64)
        scratch_M = np.zeros(shape=(count_matrices.shape[1],), dtype=np.float64)
        scratch_MM = np.zeros(shape=count_matrices.shape[1:3], dtype=np.float64)
        old_biased_conf_energies = biased_conf_energies.copy()
        old_log_lagrangian_mult = log_lagrangian_mult.copy()
        old_stat_vectors = np.zeros(shape=state_counts.shape, dtype=np.float64)
        old_therm_energies = np.zeros(shape=count_matrices.shape[0], dtype=np.float64)
        for _m in range(maxiter):
            print(sci_count)
            sci_count += 1
            tram.update_lagrangian_mult(
                old_log_lagrangian_mult, biased_conf_energies, count_matrices, state_counts,
                self.nthermo, self.nstates_full, scratch_M, log_lagrangian_mult)
            l = self.update_biased_conf_energies(
                log_lagrangian_mult, old_biased_conf_energies, count_matrices, bias_energy_sequences,
                state_sequences, state_counts, log_R_K_i, scratch_M, scratch_T, biased_conf_energies,
                scratch_MM, sci_count == save_convergence_info)

            self.therm_energies = np.zeros(shape=(self.nthermo), dtype=np.float64)
            tram.get_therm_energies(biased_conf_energies, self.nthermo, self.nstates_full,
                                    scratch_M, self.therm_energies)
            stat_vectors = np.exp(self.therm_energies[:, np.newaxis] - biased_conf_energies)
            delta_therm_energies = np.abs(self.therm_energies - old_therm_energies)
            delta_stat_vectors = np.abs(stat_vectors - old_stat_vectors)
            err = max(np.max(delta_therm_energies), np.max(delta_stat_vectors))
            if sci_count == save_convergence_info:
                sci_count = 0
                increments.append(err)
                loglikelihoods.append(l)
            # if callback is not None:
            #     try:
            #         callback(biased_conf_energies=biased_conf_energies,
            #                  log_lagrangian_mult=log_lagrangian_mult,
            #                  therm_energies=therm_energies,
            #                  stat_vectors=stat_vectors,
            #                  old_biased_conf_energies=old_biased_conf_energies,
            #                  old_log_lagrangian_mult=old_log_lagrangian_mult,
            #                  old_stat_vectors=old_stat_vectors,
            #                  old_therm_energies=old_therm_energies,
            #                  iteration_step=_m,
            #                  err=err,
            #                  maxerr=maxerr,
            #                  maxiter=maxiter)
            #     except CallbackInterrupt:
            #         break
            if err < maxerr:
                break
            else:
                shift = np.min(biased_conf_energies)
                biased_conf_energies -= shift
                old_biased_conf_energies[:] = biased_conf_energies
                old_log_lagrangian_mult[:] = log_lagrangian_mult[:]
                old_therm_energies[:] = self.therm_energies[:] - shift
                old_stat_vectors[:] = stat_vectors[:]
        conf_energies = self.get_conf_energies(bias_energy_sequences, state_sequences, log_R_K_i, scratch_T)
        tram.get_therm_energies(biased_conf_energies, self.nthermo, self.nstates_full, scratch_M, self.therm_energies)
        tram.normalize(conf_energies, biased_conf_energies, self.therm_energies, self.nthermo, self.nstates_full,
                       scratch_M)
        if err >= maxerr:
            import warnings
            warnings.warn("TRAM did not converge: last increment = %.5e" % err, UserWarning)
        if save_convergence_info == 0:
            increments = None
            loglikelihoods = None
        else:
            increments = np.array(increments, dtype=np.float64)
            loglikelihoods = np.array(loglikelihoods, dtype=np.float64)

        return biased_conf_energies, conf_energies, self.therm_energies, log_lagrangian_mult, \
               increments, loglikelihoods

    def fit(self, data, *args, **kw):
        r""" Fits a new markov state model according to data. Data may be provided clustered or non-clustered. In the
        latter case the data will be clustered first by the deeptime.clustering module. """
        # transition_matrix = np.ones((2, 2)) * 0.5
        # stationary_distribution = np.ones(2)
        #
        # res = _markov_bindings.tram.tram_test(1)
        # print(res)

        # self._model = MarkovStateModelCollection([transition_matrix], [stationary_distribution], reversible=True, count_models=[None], transition_matrix_tolerance=0)

        # TODO: if dtrajs is empty: discretize samples using some clustering algorithm
        ttrajs, dtrajs_full, btrajs = data
        # shape and type checks
        assert len(ttrajs) == len(dtrajs_full) == len(btrajs)
        for t in ttrajs:
            types.ensure_integer_array(t, ndim=1)
        for d in dtrajs_full:
            types.ensure_integer_array(d, ndim=1)
        for b in btrajs:
            types.ensure_floating_array(b, ndim=2)
        # find dimensions
        nstates_full = max(np.max(d) for d in dtrajs_full) + 1
        if self.nstates_full is None:
            self.nstates_full = nstates_full
        elif self.nstates_full < nstates_full:
            raise RuntimeError("Found more states (%d) than specified by nstates_full (%d)" % (
                nstates_full, self.nstates_full))
        self.nthermo = max(np.max(t) for t in ttrajs) + 1
        # dimensionality checks
        for t, d, b, in zip(ttrajs, dtrajs_full, btrajs):
            assert t.shape[0] == d.shape[0] == b.shape[0]
            assert b.shape[1] == self.nthermo

        # cast types and change axis order if needed
        ttrajs = [np.require(t, dtype=np.intc, requirements='C') for t in ttrajs]
        dtrajs_full = [np.require(d, dtype=np.intc, requirements='C') for d in dtrajs_full]
        btrajs = [np.require(b, dtype=np.float64, requirements='C') for b in btrajs]

        # # if equilibrium information is given, separate the trajectories
        # if self.equilibrium is not None:
        #     assert len(self.equilibrium) == len(ttrajs)
        #     _ttrajs, _dtrajs_full, _btrajs = ttrajs, dtrajs_full, btrajs
        #     ttrajs = [ttraj for eq, ttraj in zip(self.equilibrium, _ttrajs) if not eq]
        #     dtrajs_full = [dtraj for eq, dtraj in zip(self.equilibrium, _dtrajs_full) if not eq]
        #     self.btrajs = [btraj for eq, btraj in zip(self.equilibrium, _btrajs) if not eq]
        #     equilibrium_ttrajs = [ttraj for eq, ttraj in zip(self.equilibrium, _ttrajs) if eq]
        #     equilibrium_dtrajs_full = [dtraj for eq, dtraj in zip(self.equilibrium, _dtrajs_full) if eq]
        #     self.equilibrium_btrajs = [btraj for eq, btraj in zip(self.equilibrium, _btrajs) if eq]
        # else:  # set dummy values
        equilibrium_ttrajs = []
        equilibrium_dtrajs_full = []
        self.equilibrium_btrajs = []
        self.btrajs = btrajs

        # # TODO: handle RE case:
        # #  define mapping that gives for each trajectory the slices that make up a trajectory inbetween RE swaps.
        # #  At every RE swapp point, the trajectory is sliced, so that swap point occurs as a trajectory start
        # trajectory_fragment_mapping = _binding.get_RE_trajectory_fragments(ttrajs)
        # trajectory_fragments = [[dtrajs_full[tidx][start:end] for tidx, start, end in mapping_therm] for mapping_therm
        #                         in trajectory_fragment_mapping]

        # find state visits and transition counts
        state_counts_hist = [count_states(dtrajs_full[i]) for i in range(self.nthermo)]

        # histograms row sizes are equal to highest occurring markov state index.
        # Pad with zeros into a rectangular np array
        state_counts_full = np.zeros((self.nthermo, self.nstates_full))
        for idx, row in enumerate(state_counts_hist):
            state_counts_full[idx, :len(row)] += row

        # find count matrixes C^k_ij with shape (K,B,B)
        estimator = TransitionCountEstimator(lagtime=self.lagtime, count_mode=self.count_mode)
        count_matrices = [estimator.fit(dtrajs_full[i]).fetch_model().count_matrix for i in range(self.nthermo)]

        # again, histograms sizes are equal to highest occurring markov state index.
        # Pad with zeros so that all histograms are equal size
        count_matrices_full = np.zeros((self.nthermo, self.nstates_full, self.nstates_full))
        for idx, count_matrix in enumerate(count_matrices):
            count_matrices_full[idx][:len(count_matrix), :len(count_matrix)] += count_matrix

        self.therm_state_counts_full = state_counts_full.sum(axis=1)

        # if self.equilibrium is not None:
        #     self.equilibrium_state_counts_full = _util.state_counts(equilibrium_ttrajs, equilibrium_dtrajs_full,
        #                                                             nstates=self.nstates_full, nthermo=self.nthermo)
        # else:
        self.equilibrium_state_counts_full = np.zeros((self.nthermo, self.nstates_full), dtype=np.float64)

        self.csets, pcset = cset.compute_csets_TRAM(
            self.connectivity, state_counts_full, count_matrices_full,
            equilibrium_state_counts=self.equilibrium_state_counts_full,
            ttrajs=ttrajs + equilibrium_ttrajs, dtrajs=dtrajs_full + equilibrium_dtrajs_full,
            bias_trajs=self.btrajs + self.equilibrium_btrajs,
            nn=self.nn, factor=self.connectivity_factor
        )
        self.active_set = pcset

        # check for empty states
        for k in range(self.nthermo):
            if len(self.csets[k]) == 0:
                import warnings
                with warnings.catch_warnings():
                    from deeptime.util.exceptions import EmptyStateWarning
                    warnings.filterwarnings('always', message='Thermodynamic state %d' % k
                                                              + ' contains no samples after reducing to the connected set.',
                                            category=EmptyStateWarning)

        # deactivate samples not in the csets, states are *not* relabeled
        self.state_counts, self.count_matrices, self.dtrajs, _ = cset.restrict_to_csets(
            self.csets,
            state_counts=state_counts_full, count_matrices=count_matrices_full,
            ttrajs=ttrajs, dtrajs=dtrajs_full)

        if self.equilibrium is not None:
            self.equilibrium_state_counts, _, self.equilibrium_dtrajs, _ = cset.restrict_to_csets(
                self.csets,
                state_counts=self.equilibrium_state_counts_full, ttrajs=equilibrium_ttrajs,
                dtrajs=equilibrium_dtrajs_full)
        else:
            self.equilibrium_state_counts = np.zeros((self.nthermo, self.nstates_full),
                                                     dtype=np.intc)  # (remember: no relabeling)
            self.equilibrium_dtrajs = []

        # self-consistency tests
        assert np.all(self.state_counts >= np.maximum(self.count_matrices.sum(axis=1), \
                                                      self.count_matrices.sum(axis=2)))
        assert np.all(np.sum(
            [np.bincount(d[d >= 0], minlength=self.nstates_full) for d in self.dtrajs],
            axis=0) == self.state_counts.sum(axis=0))
        assert np.all(np.sum(
            [np.bincount(t[d >= 0], minlength=self.nthermo) for t, d in zip(ttrajs, self.dtrajs)],
            axis=0) == self.state_counts.sum(axis=1))
        if self.equilibrium is not None:
            assert np.all(np.sum(
                [np.bincount(d[d >= 0], minlength=self.nstates_full) for d in self.equilibrium_dtrajs],
                axis=0) == self.equilibrium_state_counts.sum(axis=0))
            assert np.all(np.sum(
                [np.bincount(t[d >= 0], minlength=self.nthermo) for t, d in
                 zip(equilibrium_ttrajs, self.equilibrium_dtrajs)],
                axis=0) == self.equilibrium_state_counts.sum(axis=1))

        # check for empty states
        for k in range(self.state_counts.shape[0]):
            if self.count_matrices[k, :, :].sum() == 0 and self.equilibrium_state_counts[k, :].sum() == 0:
                import warnings
                with warnings.catch_warnings():
                    from deeptime.util.exceptions import EmptyStateWarning
                    warnings.filterwarnings('always', message='Thermodynamic state %d' % k \
                                                              + ' contains no transitions and no equilibrium data after reducing to the connected set.',
                                            category=EmptyStateWarning)

        # if self.init == 'mbar' and self.biased_conf_energies is None:
        #     # if self.direct_space:
        #     #     mbar = _mbar_direct
        #     # else:
        #     #     mbar = _mbar
        #     stage = 'MBAR init.'
        #     self.mbar_therm_energies, self.mbar_unbiased_conf_energies, \
        #         self.mbar_biased_conf_energies, _ = mbar.estimate(
        #             (state_counts_full.sum(axis=1) + self.equilibrium_state_counts_full.sum(axis=1)).astype(_np.intc),
        #             self.btrajs + self.equilibrium_btrajs, dtrajs_full + equilibrium_dtrajs_full,
        #             maxiter=self.init_maxiter, maxerr=self.init_maxerr,
        #             n_conf_states=self.nstates_full)
        #     self.biased_conf_energies = self.mbar_biased_conf_energies.copy()

        # run estimator
        # if self.direct_space:
        #     tram = _tram_direct
        #     trammbar = _trammbar_direct
        # else:
        #     tram = _tram
        #     trammbar = _trammbar
        # import warnings
        # with warnings.catch_warnings() as cm:
        # warnings.filterwarnings('ignore', RuntimeWarning)
        stage = 'TRAM'
        if self.equilibrium is None:
            self.biased_conf_energies, conf_energies, self.therm_energies, self.log_lagrangian_mult, \
            self.increments, self.loglikelihoods = self.fit_from_count_matrices(
                self.count_matrices, self.state_counts, self.btrajs, self.dtrajs,
                maxiter=self.maxiter, maxerr=self.maxerr,
                biased_conf_energies=self.biased_conf_energies,
                log_lagrangian_mult=self.log_lagrangian_mult,
                save_convergence_info=self.save_convergence_info)
        # else:  # use trammbar
        #     self.biased_conf_energies, conf_energies, self.therm_energies, self.log_lagrangian_mult, \
        #     self.increments, self.loglikelihoods = trammbar.estimate(
        #         self.count_matrices, self.state_counts, self.btrajs, self.dtrajs,
        #         equilibrium_therm_state_counts=self.equilibrium_state_counts.sum(axis=1).astype(_np.intc),
        #         equilibrium_bias_energy_sequences=self.equilibrium_btrajs,
        #         equilibrium_state_sequences=self.equilibrium_dtrajs,
        #         maxiter=self.maxiter, maxerr=self.maxerr,
        #         save_convergence_info=self.save_convergence_info,
        #         biased_conf_energies=self.biased_conf_energies,
        #         log_lagrangian_mult=self.log_lagrangian_mult,
        #         callback=_ConvergenceProgressIndicatorCallBack(
        #             pg, stage, self.maxiter, self.maxerr, subcallback=self.callback),
        #         N_dtram_accelerations=self.N_dtram_accelerations,
        #         overcounting_factor=self.overcounting_factor)

        # compute models
        # fmsms = [np.ascontiguousarray((
        #                                    tram.estimate_transition_matrix(
        #                                        self.log_lagrangian_mult, self.biased_conf_energies, self.count_matrices,
        #                                        None,
        #                                        K)[self.active_set, :])[:, self.active_set]) for K in
        #          range(self.nthermo)]
        #
        # active_sets = [compute_connected_sets(msm, directed=False)[0] for msm in fmsms]
        # fmsms = [np.ascontiguousarray(
        #     (msm[lcc, :])[:, lcc]) for msm, lcc in zip(fmsms, active_sets)]
        #
        # models = []
        # for i, (msm, acs) in enumerate(zip(fmsms, active_sets)):
        #     pi_acs = np.exp(self.therm_energies[i] - self.biased_conf_energies[i, :])[self.active_set[acs]]
        #     pi_acs = pi_acs / pi_acs.sum()
        #     # models.append(_ThermoMSM(
        #     #     msm, self.active_set[acs], self.nstates_full, pi=pi_acs,
        #     #     dt_model=self.timestep_traj.get_scaled(self.lag)))
        #
        # # set model parameters to self
        # self.set_model_params(
        #     models=models, f_therm=self.therm_energies, f=conf_energies[self.active_set].copy())

        return self

    # TODO: move to tram.h
    def update_biased_conf_energies(self, log_lagrangian_mult, biased_conf_energies, count_matrices,
                                    bias_energy_sequences, state_sequences, state_counts, log_R_K_i,
                                    scratch_M, scratch_T, new_biased_conf_energies, scratch_MM, return_log_L=False):
        new_biased_conf_energies[:] = np.inf

        tram.get_log_Ref_K_i(log_lagrangian_mult, biased_conf_energies,
                             count_matrices, state_counts, log_lagrangian_mult.shape[0],
                             log_lagrangian_mult.shape[1], scratch_M, log_R_K_i)
        log_L = 0.0

        for i in range(len(bias_energy_sequences)):
            log_L += tram.update_biased_conf_energies(bias_energy_sequences[i], state_sequences[i],
                                                      state_sequences[i].shape[0], log_R_K_i,
                                                      log_lagrangian_mult.shape[0],
                                                      log_lagrangian_mult.shape[1],
                                                      scratch_T, new_biased_conf_energies, int(return_log_L))
        if return_log_L:
            assert scratch_MM is not None
            log_L += tram.discrete_log_likelihood_lower_bound(log_lagrangian_mult, new_biased_conf_energies,
                                                              count_matrices, state_counts, state_counts.shape[0],
                                                              state_counts.shape[1], scratch_M, scratch_MM)
            return log_L

        # TODO: move to tram.h

    def get_conf_energies(self, bias_energy_sequences, state_sequences, log_R_K_i, scratch_T):
        r"""
        Update the reduced unbiased free energies
        Parameters
        ----------
        bias_energy_sequences : list of numpy.ndarray(shape=(X_i, T), dtype=numpy.float64)
            reduced bias energies in the T thermodynamic states for all X samples
        state_sequence : list of numpy.ndarray(shape=(X_i,), dtype=numpy.intc)
            Markov state indices for all X samples
        log_R_K_i : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
            precomputed sum of TRAM log pseudo-counts and biased_conf_energies
        scratch_T : numpy.ndarray(shape=(T), dtype=numpy.float64)
            scratch array for logsumexp operations
        Returns
        -------
        conf_energies : numpy.ndarray(shape=(M,), dtype=numpy.float64)
            unbiased (Markov) free energies
        """
        conf_energies = np.zeros(shape=(log_R_K_i.shape[1],), dtype=np.float64)
        conf_energies[:] = np.inf
        for i in range(len(bias_energy_sequences)):
            tram.get_conf_energies(bias_energy_sequences[i], state_sequences[i], state_sequences[i].shape[0],
                                   log_R_K_i, self.nthermo, self.nstates_full, scratch_T, conf_energies)
        return conf_energies
