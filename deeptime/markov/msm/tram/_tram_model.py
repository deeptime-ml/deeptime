import numpy as np

from deeptime.numeric._numeric_bindings import logsumexp
from deeptime.markov.msm import MarkovStateModelCollection
from deeptime.base import Model
from ._tram_bindings import tram


class TRAMModel(Model):
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

        self._markov_state_model_collection = self._construct_markov_model_collection(
            count_models, transition_matrices)

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
    def markov_state_model_collection(self):
        r""" The underlying MarkovStateModelCollection. Contains one Markov state model for each sampled thermodynamic
        state.

        Returns
        -------
        markov_state_model_collection : MarkovStateModelCollection
            the collection of markov state models containing one model for each thermodynamic state.
        """
        return self._markov_state_model_collection

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

    def compute_observable(self, dtrajs, bias_matrices, observable_values, therm_state=-1):
        sample_weights = self.compute_sample_weights(dtrajs, bias_matrices, therm_state)

        # flatten both
        sample_weights = np.reshape(sample_weights, -1)
        observable_values = np.reshape(observable_values, -1)

        return np.dot(sample_weights, observable_values)

    def compute_PMF(self, dtrajs, bias_matrices, binned_samples, therm_state=-1, n_bins=None):
        # TODO: account for variable bin widths
        sample_weights = np.reshape(self.compute_sample_weights(dtrajs, bias_matrices, therm_state), -1)
        binned_samples = np.reshape(binned_samples, -1)

        if n_bins is None:
            n_bins = binned_samples.max()

        pmf = np.zeros(n_bins)

        for i in range(len(pmf)):
            indices = np.where(binned_samples == i)
            pmf[i] = -np.log(np.sum(sample_weights[indices]))

        # shift minimum to zero
        pmf -= pmf.min()
        return pmf

    def _construct_markov_model_collection(self, count_models, transition_matrices):
        r""" Construct a MarkovStateModelCollection from the transition matrices and energy estimates.
        For each of the thermodynamic states, one MarkovStateModel is added to the MarkovStateModelCollection. The
        corresponding count models are previously calculated and are restricted to the largest connected set.

        Returns
        -------
        markov_state_model_collection : MarkovStateModelCollection
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
