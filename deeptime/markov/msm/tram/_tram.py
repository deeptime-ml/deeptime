from typing import Optional

from deeptime.markov.msm import MarkovStateModelCollection
from deeptime.markov import TransitionCountEstimator
from deeptime.markov._base import _MSMBaseEstimator
from deeptime.util import types
from deeptime.markov import _tram_bindings
from deeptime.markov import _markov_bindings, compute_connected_sets
from ._cset import *

import numpy as np


def _to_zero_padded_array(array_list, desired_shape):
    for i, array in enumerate(array_list):
        pad_amount = [(0, diff) for diff in np.asarray(desired_shape) - np.asarray(array.shape)]
        array_list[i] = np.pad(array, pad_amount, 'constant')
    return np.ascontiguousarray(array_list)


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
            nn=None, connectivity_threshold: float = 1.0):
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
        connectivity_threshold : float, optional, default=1.0
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
        self.connectivity_threshold = connectivity_threshold
        self.n_markov_states = None
        self.n_therm_states = None
        self.maxiter = maxiter
        self.maxerr = maxerr
        self.save_convergence_info = save_convergence_info
        self.active_set = None

        self.counts_models = []
        self.tram_estimator = None

    @property
    def therm_state_energies(self) -> Optional:
        if self.tram_estimator is not None:
            return self.tram_estimator.therm_state_energies()
        return None

    @property
    def biased_conf_energies(self):
        if self.tram_estimator is not None:
            return self.tram_estimator.biased_conf_energies()

    @property
    def markov_state_energies(self):
        if self.tram_estimator is not None:
            return self.tram_estimator.markov_state_energies()

    def fetch_model(self) -> Optional[MarkovStateModelCollection]:
        r"""Yields the most recent :class:`MarkovStateModelCollection` that was estimated.
        Can be None if fit was not called.

        Returns
        -------
        model : MarkovStateModelCollection or None
            The most recent markov state model or None.
        """
        return self._model

    def fit(self, data, *args, **kw):
        r""" Fits a new markov state model according to data. Data may be provided clustered or non-clustered. In the
        latter case the data will be clustered first by the deeptime.clustering module. """
        # TODO: allow for empty ttrajs
        therm_state_sequences, dtrajs, bias_matrix = self._validate_input(data)

        # count all transitions and state counts, without restricting to connected sets
        self.counts_models = self._get_state_counts(dtrajs)

        # restrict input data to largest connected set
        transition_counts, state_counts, dtrajs = self._restrict_to_connected_set(dtrajs)

        # TODO: make progress bar
        def callback(iteration, error, log_likelihood):
            print(f"Iteration {iteration}: error {error}. log_L: {log_likelihood}")

        tram_input = _tram_bindings.TRAM_input(state_counts, transition_counts, dtrajs, bias_matrix)
        self.tram_estimator = _tram_bindings.TRAM(tram_input)
        self.tram_estimator.estimate(10, np.float64(1e-8), True, callback)

        self._to_markov_model()

        return self

    def _to_markov_model(self):
        # compute models
        transition_matrices = self.tram_estimator.transition_matrices()

        transition_matrices_connected = []
        stationary_distributions = []
        for i, msm in enumerate(transition_matrices):
            states = self.counts_models[i].states
            transition_matrices_connected.append(transition_matrices[i][states][:, states])
            pi = np.exp(self.therm_state_energies[i] - self.biased_conf_energies[i, :])[states]
            pi = pi / pi.sum()
            stationary_distributions.append(pi)

        self._model = MarkovStateModelCollection(transition_matrices_connected, stationary_distributions,
                                                 reversible=True, count_models=self.counts_models,
                                                 transition_matrix_tolerance=1e-8)

    def _validate_input(self, data):
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

    def _get_state_counts(self, dtrajs):
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

        # find count matrixes C^k_ij with shape (K,B,B)
        estimator = TransitionCountEstimator(lagtime=self.lagtime, count_mode=self.count_mode)

        count_models = []

        for i in range(self.n_therm_states):
            counts = estimator.fit_fetch(dtrajs[i])
            count_models.append(counts)

        return count_models

    def _restrict_to_connected_set(self, dtrajs):
        transition_counts_list = []
        state_counts_list = []
        dtrajs_connected = []

        for i in range(self.n_therm_states):
            # submodel is the largest connected set
            submodel = self.counts_models[i].submodel_largest(connectivity_threshold=self.connectivity_threshold,
                                                              directed=True)

            # Assign -1 to all indices not in the submodel.
            restricted_dtraj = submodel.transform_discrete_trajectories_to_submodel(dtrajs[i])

            # The restricted_dtraj indices are those that belong to the submodel states. We don't want that, we want the
            # original state indices (and those -1s). Convert the newly assigned indices back to the original state
            # symbols
            restricted_dtraj_symb = submodel.states_to_symbols(restricted_dtraj)

            dtrajs_connected.append(restricted_dtraj_symb)
            transition_counts_list.append(submodel.count_matrix.astype(np.intc))
            state_counts_list.append(submodel.state_histogram.astype(np.intc))

        # for TRAM, all count matrices need to be the same size. Pad the smaller ones (that were restricted) with zeros.
        state_counts = _to_zero_padded_array(state_counts_list, self.nstates_full)
        transition_counts = _to_zero_padded_array(transition_counts_list, (self.nstates_full, self.nstates_full))

        return transition_counts, state_counts, dtrajs_connected
