from typing import Optional

from deeptime.markov.msm import MarkovStateModelCollection
from deeptime.markov import TransitionCountEstimator, count_states
from deeptime.markov._base import _MSMBaseEstimator
from deeptime.util import types
from deeptime.markov import _tram_bindings
from deeptime.markov import _markov_bindings, compute_connected_sets
from ._cset import *

import numpy as np


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
            maxiter=10000, maxerr: float = 1.0E-15, save_convergence_info=0,
            nn=None, connectivity_factor: float = 1.0):
        r"""Transition(-based) Reweighting Analysis Method
        Parameters
        ----------
        lagtime : int
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
        maxiter : int, optional, default=10000
            The maximum number of self-consistent iterations before the estimator exits unsuccessfully.
        maxerr : float, optional, default=1E-15
            Convergence criterion based on the maximal free energy change in a self-consistent
            iteration step.
        save_convergence_info : int, optional, default=0
            Every saveConvergenceInfo iteration steps, store the actual increment
            and the actual log-likelihood; 0 means no storage.
        connectivity_factor : float, optional, default=1.0
            Only needed if connectivity='post_hoc_RE' or 'BAR_variance'. Values
            greater than 1.0 weaken the connectivity conditions. For 'post_hoc_RE'
            this multiplies the number of hypothetically observed transitions. For
            'BAR_variance' this scales the threshold for the minimal allowed variance
            of free energy differences.

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
        self.n_markov_states = None
        self.n_therm_states = None
        self.maxiter = maxiter
        self.maxerr = maxerr
        self.save_convergence_info = save_convergence_info
        self.active_set = None

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

    def fit(self, data, *args, **kw):
        r""" Fits a new markov state model according to data. Data may be provided clustered or non-clustered. In the
        latter case the data will be clustered first by the deeptime.clustering module. """
        # TODO: if markov_state_sequences_full is empty: discretize samples using some clustering algorithm

        therm_state_sequences_full, dtrajs_full, bias_matrix = self._check_data(data)

        # count all transitions and state counts, without restricting to connected sets
        state_counts_full, transition_counts_models = self._get_state_counts(dtrajs_full)

        transition_counts_full = self._to_padded_transition_count_matrix(transition_counts_models)

        # restrict input data to connected set
        state_counts, transition_counts, dtrajs = \
            self._restrict_to_connected_sets(therm_state_sequences_full, dtrajs_full, bias_matrix,
                                             state_counts_full, transition_counts_full)

        #TODO: user should provide this.
        def callback(iteration, error):
            print(f"Iteration {iteration}: error {error}")

        print("INSTANTIATING TRAM INPUT...")
        tram_input = _tram_bindings.TRAM_input(state_counts, transition_counts, dtrajs, bias_matrix)
        print("INSTANTIATING TRAM...")
        tram = _tram_bindings.TRAM(tram_input)
        print("STARTING ESTIMATION...")
        tram.estimate(self.maxiter, np.float64(1e-8), callback)
        print("TRAM ESTIMATION DONE.")
        self.biased_conf_energies = tram.biased_conf_energies()
        return self
        # compute models
        transition_matrices = tram.estimate_transition_matrices()

        active_sets = [compute_connected_sets(msm, directed=False)[0] for msm in transition_matrices]
        transition_matrices = [np.ascontiguousarray(
            (msm[lcc, :])[:, lcc]) for msm, lcc in zip(transition_matrices, active_sets)]

        stationary_distributions = []
        for i, (msm, acs) in enumerate(zip(transition_matrices, active_sets)):
            pi_acs = np.exp(self.therm_energies[i] - self.biased_conf_energies[i, :])[self.active_set[acs]]
            pi_acs = pi_acs / pi_acs.sum()
            stationary_distributions.append(pi_acs)

        self._model = MarkovStateModelCollection(transition_matrices, stationary_distributions, reversible=True,
                                                 count_models=transition_counts_models,
                                                 transition_matrix_tolerance=1e-8)
        return self

    def _check_data(self, data):
        therm_state_sequences, dtrajs, bias_energy_sequences = data

        # shape and type checks
        assert len(therm_state_sequences) == len(dtrajs) == len(bias_energy_sequences)
        for t in therm_state_sequences:
            types.ensure_integer_array(t, ndim=1)
        for d in dtrajs:
            types.ensure_integer_array(d, ndim=1)
        for b in bias_energy_sequences:
            types.ensure_floating_array(b, ndim=2)

        # find dimensions
        self.nstates_full = max(np.max(d) for d in dtrajs) + 1
        self.n_therm_states = max(np.max(t) for t in therm_state_sequences) + 1

        # dimensionality checks
        for t, d, b, in zip(therm_state_sequences, dtrajs, bias_energy_sequences):
            assert t.shape[0] == d.shape[0] == b.shape[0]
            assert b.shape[1] == self.n_therm_states

        # cast types and change axis order if needed
        therm_state_sequences = [np.require(t, dtype=np.intc, requirements='C') for t in therm_state_sequences]
        markov_state_sequences = [np.require(d, dtype=np.intc, requirements='C') for d in dtrajs]
        bias_energy_sequences = [np.require(b, dtype=np.float64, requirements='C') for b in bias_energy_sequences]

        return therm_state_sequences, markov_state_sequences, bias_energy_sequences

    def _get_state_counts(self, dtrajs_full):
        """ Get transition counts and state counts for the discrete trajectories. """
        # find state visits and transition counts
        # TODO:
        #  1. handle RE case:
        #     define mapping that gives for each trajectory the slices that make up a trajectory inbetween RE swaps.
        #     At every RE swapp point, the trajectory is sliced, so that swap point occurs as a trajectory start
        #  2. Include therm_state_sequences_full --> don't assume that each dtraj[i] was sampled at thermodynamc state i.
        # for 1.: do something like this
        # trajectory_fragment_mapping = _binding.get_RE_trajectory_fragments(therm_state_sequences_full)
        # trajectory_fragments = [[markov_state_sequences[tidx][start:end] for tidx, start, end in mapping_therm]
        #                               for mapping_therm in trajectory_fragment_mapping]

        state_counts = np.ascontiguousarray(
            [count_states(dtrajs_full[i]) for i in range(self.n_therm_states)])

        # find count matrixes C^k_ij with shape (K,B,B)
        estimator = TransitionCountEstimator(lagtime=self.lagtime, count_mode=self.count_mode)
        transition_counts_models = [estimator.fit(dtrajs_full[i]).fetch_model() for i in
                                    range(self.n_therm_states)]

        return state_counts, transition_counts_models

    def _to_padded_transition_count_matrix(self, transition_counts_models):
        """ Transform an array of transition count models into a 3D-matrix containing the transition counts
        for each transition count model. The transition count models need not have transition count matrices of
        the same size; count matrices are padded with zeros from the right."""

        # transition matrix sizes for each state are equal to highest occurring markov state index.
        transition_counts = [model.count_matrix for model in transition_counts_models]

        # Fill an (KxBxB) array so that all transition matrices are equal sized.
        transition_counts_padded = np.zeros((self.n_therm_states, self.nstates_full, self.nstates_full))
        for idx, count_matrix in enumerate(transition_counts):
            transition_counts_padded[idx][:len(count_matrix), :len(count_matrix)] += count_matrix

        return transition_counts_padded

    def _restrict_to_connected_sets(self, therm_state_sequences_full, markov_state_sequences_full, bias_matrix,
                                    state_counts_full, transition_counts_full):
        """ restict input trajectories to only contain samples that are in the connected sets. """
        csets, pcset = compute_csets_TRAM(
            self.connectivity, state_counts_full, transition_counts_full,
            ttrajs=therm_state_sequences_full, dtrajs=markov_state_sequences_full,
            bias_trajs=bias_matrix,
            nn=self.nn, factor=self.connectivity_factor
        )
        self.active_set = pcset

        self._check_for_empty_csets(csets)

        # deactivate samples not in the csets, states are *not* relabeled
        state_counts, transition_counts, markov_state_sequences, _ = restrict_to_csets(
            csets, state_counts=state_counts_full, count_matrices=transition_counts_full,
            ttrajs=therm_state_sequences_full, dtrajs=markov_state_sequences_full)

        # self-consistency tests
        assert np.all(state_counts >= np.maximum(transition_counts.sum(axis=1),
                                                 transition_counts.sum(axis=2)))
        assert np.all(np.sum(
            [np.bincount(d[d >= 0], minlength=self.nstates_full) for d in markov_state_sequences],
            axis=0) == state_counts.sum(axis=0))
        assert np.all(np.sum(
            [np.bincount(t[d >= 0], minlength=self.n_therm_states) for t, d in
             zip(therm_state_sequences_full, markov_state_sequences)],
            axis=0) == state_counts.sum(axis=1))

        self._check_for_states_without_transitions(transition_counts)

        return state_counts, transition_counts, markov_state_sequences

    def _check_for_empty_csets(self, csets):
        # check for empty states
        for k in range(self.n_therm_states):
            if len(csets[k]) == 0:
                import warnings
                with warnings.catch_warnings():
                    from deeptime.util.exceptions import EmptyStateWarning
                    warnings.filterwarnings('always', message='Thermodynamic state %d' % k
                                                              + ' contains no samples after reducing to the connected set.',
                                            category=EmptyStateWarning)

    def _check_for_states_without_transitions(self, transition_counts):
        # check for empty states
        for k in range(self.n_therm_states):
            if transition_counts[k, :, :].sum() == 0:
                import warnings
                with warnings.catch_warnings():
                    from deeptime.util.exceptions import EmptyStateWarning
                    warnings.filterwarnings('always', message='Thermodynamic state %d' % k \
                                                              + ' contains no transitions after reducing to the connected set.',
                                            category=EmptyStateWarning)
