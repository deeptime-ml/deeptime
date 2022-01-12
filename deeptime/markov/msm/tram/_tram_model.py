import numpy as np

from deeptime.numeric import logsumexp
from deeptime.markov.msm import MarkovStateModelCollection
from deeptime.base import Model
from ._tram_bindings import tram


class TRAMModel(Model):
    r""" The TRAM model containing the estimated parameters, free energies, and the underlying Markov models for each
    thermodynamic state. TRAM is described in :footcite:`wu2016multiensemble`.

    Parameters
    ----------
    count_models : list(TransitionCountModel)
        The transition count models for all thermodynamic states. Have been restricted to the largest connected set.
    transition_matrices : ndarray(n, m, m), float64
        The estimated transition matrices for each thermodynamic state. The transition matrices and count models are
        combined into a MarkovStateModelCollection that holds the Markov models for each thermodynamic state.
    biased_conf_energies : ndarray(n, m), float64
        The estimated free energies :math:`f_i^k` of each Markov state for all thermodynamic states.
        biased_conf_energies[k,i] contains the bias energy of Markov state i in thermodynamic state k.
    lagrangian_mult_log : ndarray(n, m), float64
        The estimated logarithm of the lagrange multipliers :math:`v_i^k` of each Markov state for all thermodynamic
        states. lagrangian_mult_log[k,i] contains the lagrange multiplier of Markov state i in thermodynamic state k.
    modified_state_counts_log : ndarray(n, m), float64
        The logarithm of the modified state counts :math:`R_i^k` of each Markov state for all thermodynamic
        states. modified_state_counts_log[k,i] contains the state counts of Markov state i in thermodynamic state k.
    therm_state_energies : ndarray(n), float64
        The estimated free energy of each thermodynamic state, :math:`f^k`.
    markov_state_energies : ndarray(m), float64
        The estimated free energy of each Markov state, :math: `f_i`.

    References
    ----------
    .. footbibliography::
    """
    def __init__(self, count_models, transition_matrices,
                 biased_conf_energies,
                 lagrangian_mult_log,
                 modified_state_counts_log,
                 therm_state_energies=None,
                 markov_state_energies=None,
                 ):
        self.n_therm_states = biased_conf_energies.shape[0]
        self.n_markov_states = biased_conf_energies.shape[1]

        self._biased_conf_energies = biased_conf_energies
        self._modified_state_counts_log = modified_state_counts_log
        self._lagrangian_mult_log = lagrangian_mult_log
        self._markov_state_energies = markov_state_energies

        if therm_state_energies is None:
            self._therm_state_energies = np.asarray([-logsumexp(-row) for row in biased_conf_energies])
        else:
            self._therm_state_energies = therm_state_energies

        self._msm_collection = self._construct_msm_collection(
            count_models, transition_matrices)

    @property
    def biased_conf_energies(self) -> np.ndarray:
        r""" The estimated free energy per thermodynamic state and Markov state, :math:`f_k^i`, where :math:`k` is the
        thermodynamic state index, and :math:`i` the Markov state index.
        """
        return self._biased_conf_energies

    @property
    def lagrangian_mult_log(self) -> np.ndarray:
        r""" The estimated logarithm of the lagrange multipliers :math:`v_i^k` of each Markov state for all thermodynamic
            states. lagrangian_mult_log[k,i] contains the lagrange multiplier of Markov state i in thermodynamic state k.
        """
        return self._lagrangian_mult_log

    @property
    def modified_state_counts_log (self):
        r"""The logarithm of the modified state counts :math:`R_i^k` of each Markov state for all thermodynamic
        states. modified_state_counts_log[k,i] contains the state counts of Markov state i in thermodynamic state k.
        """
        return self._modified_state_counts_log

    @property
    def therm_state_energies(self) -> np.ndarray:
        r""" The estimated free energy per thermodynamic state, :math:`f_k`, where :math:`k` is the thermodynamic state
        index.
        """
        return self._therm_state_energies


    @property
    def markov_state_energies(self) -> np.ndarray:
        r""" The estimated free energy per Markov state, :math:`f^i`, where :math:`i` is the Markov state
        index.
        """
        return self._markov_state_energies

    @property
    def msm_collection(self):
        r""" The underlying MarkovStateModelCollection. Contains one Markov state model for each sampled thermodynamic
        state.

        :getter: The collection of markov state models containing one model for each thermodynamic state.
        :type: MarkovStateModelCollection
        """
        return self._msm_collection

    def compute_sample_weights(self, dtrajs, bias_matrices, therm_state=-1):
        r""" Compute the sample weight :math:`\mu(x)` for all samples :math:`x`. If the thermodynamic state index is >=0,
        the sample weights for that thermodynamic state will be computed, i.e. :math:`\mu^k(x)`. Otherwise, this gives
        the unbiased sample weights.

        Parameters
        ----------
        therm_state : int, optional
            The index of the thermodynamic state in which the sample weights need to be computed. If therm_state=-1,
            the unbiased sample weights are computed.

        Returns
        -------
        sample_weights : np.ndarray
            The statistical weight :math:`\mu(x)` of each sample (i.e., the probability distribution over all samples:
            the sum over all sample weights equals one.)

        Notes
        -----
        The statistical distribution is given by

        .. math:: \mu(x) = \left( \sum_k R^k_{i(x)} \mathrm{exp}[f^k_{i(k)}-b^k(x)] \right)^{-1}
        """
        return tram.compute_sample_weights(therm_state, dtrajs, bias_matrices, self._therm_state_energies,
                                           self._modified_state_counts_log)

    def compute_observable(self, observable_values, dtrajs, bias_matrices, therm_state=-1):
        r""" Compute an observable value.

        Parameters
        ----------
        observable_values : list(np.ndarray)
            The list of observable values. observable_values[i][n] contains the observable value for the n-th sample in
            the i-th trajectory.
        dtrajs : list(np.ndarray)
            The list of discrete trajectories. dtrajs[i][n] contains the Markov state index of the n-th sample in the
            i-th trajectory.
        bias_matrices : list(np.ndarray)
            The bias energy matrices. bias_matrices[i, n, l] contains the bias energy of the n-th sample from the i-th
            trajectory, evaluated at thermodynamic state k. The bias energy matrices should have the same size as
            dtrajs in both the 0-th and 1-st dimension. The seconds dimension of of size n_therm_state, i.e. for each
            sample, the bias energy in every thermodynamic state is calculated and stored in the bias_matrices.

        The observed values, bias matrices and dtrajs are associated, i.e. they all pertain to the same samples:
        for the n-th sample in the i-the trajectory, its observable, bias values and Markov state can be found
        at index n, i of observable_values, bias_matrices and dtrajs respectively.
        """
        sample_weights = self.compute_sample_weights(dtrajs, bias_matrices, therm_state)

        # flatten both
        sample_weights = np.reshape(sample_weights, -1)
        observable_values = np.reshape(observable_values, -1)

        return np.dot(sample_weights, observable_values)

    def compute_PMF(self, dtrajs, bias_matrices, bin_indices, therm_state=-1, n_bins=None):
        r""" Compute an observable value.

        Parameters
        ----------
        dtrajs : list(np.ndarray)
            The list of discrete trajectories. dtrajs[i][n] contains the Markov state index of the n-th sample in the
            i-th trajectory.
        bias_matrices : list(np.ndarray)
            The bias energy matrices. bias_matrices[i, n, l] contains the bias energy of the n-th sample from the i-th
            trajectory, evaluated at thermodynamic state k. The bias energy matrices should have the same size as
            dtrajs in both the 0-th and 1-st dimension. The seconds dimension of of size n_therm_state, i.e. for each
            sample, the bias energy in every thermodynamic state is calculated and stored in the bias_matrices.
        bin_indices : list(np.ndarray)
            The list of bin indices that the samples are binned into. The PMF is calculated as a distribution over these
            bins. binnes_samples[i][n] contains the bin index for the n-th sample in the i-th trajectory.

        The bin indices, bias matrices and dtrajs are associated, i.e. they all pertain to the same samples:
        for the n-th sample in the i-the trajectory, its bin index, bias values and Markov state can be found
        at index n, i of bin_indices, bias_matrices and dtrajs respectively.
        """
        # TODO: account for variable bin widths
        sample_weights = np.reshape(self.compute_sample_weights(dtrajs, bias_matrices, therm_state), -1)
        binned_samples = np.reshape(bin_indices, -1)

        if n_bins is None:
            n_bins = binned_samples.max() + 1

        pmf = np.zeros(n_bins)

        for i in range(len(pmf)):
            indices = np.where(binned_samples == i)
            pmf[i] = -np.log(np.sum(sample_weights[indices]))

        # shift minimum to zero
        pmf -= pmf.min()
        return pmf

    def _construct_msm_collection(self, count_models, transition_matrices):
        r""" Construct a MarkovStateModelCollection from the transition matrices and energy estimates.
        For each of the thermodynamic states, one MarkovStateModel is added to the MarkovStateModelCollection. The
        corresponding count models are previously calculated and are restricted to the largest connected set.

        Returns
        -------
        msm_collection : MarkovStateModelCollection
            the collection of markov state models containing one model for each thermodynamic state.
        """
        transition_matrices_connected = []
        stationary_distributions = []

        for i, msm in enumerate(transition_matrices):
            states = count_models[i].states
            transition_matrices_connected.append(transition_matrices[i][states][:, states])
            pi = np.exp(self.therm_state_energies[i] - self._biased_conf_energies[i, :])[states]
            pi = pi / pi.sum()
            stationary_distributions.append(pi)

        return MarkovStateModelCollection(transition_matrices_connected, stationary_distributions,
                                          reversible=True, count_models=count_models,
                                          transition_matrix_tolerance=1e-8)
