import numpy as np

from deeptime.base import Model
from deeptime.numeric import logsumexp
from deeptime.markov.msm import MarkovStateModelCollection

from ._tram_dataset import transition_counts_from_count_models, state_counts_from_count_models
from ._tram_bindings import tram


class TRAMModel(Model):
    r""" The TRAM model containing the estimated parameters, free energies, and the underlying Markov models for each
    thermodynamic state.
    The TRAM Model is the result of a TRAM estimation and can be used to calculate observables and recover an (unbiased)
    potential of mean force (PMF). TRAM is described in :footcite:`wu2016multiensemble`.

    Parameters
    ----------
    count_models : list(TransitionCountModel)
        The transition count models for all thermodynamic states.
    transition_matrices : ndarray(n, m, m), float64
        The estimated transition matrices for each thermodynamic state. The transition matrices and count models are
        combined into a MarkovStateModelCollection that holds the Markov models for each thermodynamic state.
    biased_conf_energies : ndarray(n, m), float64
        The estimated free energies :math:`f_i^k` of each Markov state for all thermodynamic states.
        `biased_conf_energies[k,i]` contains the bias energy of Markov state :math:`i` in thermodynamic state :math:`k`.
    lagrangian_mult_log : ndarray(n, m), float64
        The estimated logarithm of the lagrange multipliers :math:`v_i^k` of each Markov state for all thermodynamic
        states. `lagrangian_mult_log[k,i]` contains the lagrange multiplier of Markov state :math:`i` in thermodynamic
        state :math:`k`.
    modified_state_counts_log : ndarray(n, m), float64
        The logarithm of the modified state counts :math:`R_i^k` of each Markov state for all thermodynamic
        states. `modified_state_counts_log[k,i]` contains the state counts of Markov state :math:`i` in thermodynamic
        state :math:`k`.
    therm_state_energies : ndarray(n), float64
        The estimated free energy of each thermodynamic state, :math:`f^k`.
    markov_state_energies : ndarray(m), float64
        The estimated free energy of each Markov state, :math:`f_i`.

    See also
    --------
    :class:`MarkovStateModelCollection <deeptime.markov.msm.MarkovStateModelCollection>`,
    :class:`TRAM <deeptime.markov.msm.TRAM>`, :class:`TRAMDataset <deeptime.markov.msm.TRAMDataset>`

    References
    ----------
    .. footbibliography::
    """

    def __init__(self, count_models, transition_matrices,
                 biased_conf_energies,
                 lagrangian_mult_log,
                 modified_state_counts_log,
                 therm_state_energies=None,
                 markov_state_energies=None
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

        self._transition_matrices = transition_matrices
        self._count_models = count_models

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
        r""" The estimated logarithm of the lagrange multipliers :math:`v_i^k` of each Markov state for all
        thermodynamic states. `lagrangian_mult_log[k,i]` contains the lagrange multiplier of Markov state :math:`i` in
        thermodynamic state :math:`k`.
        """
        return self._lagrangian_mult_log

    @property
    def modified_state_counts_log(self):
        r"""The logarithm of the modified state counts :math:`R_i^k` of each Markov state for all thermodynamic
        states. `modified_state_counts_log[k,i]` contains the state counts of Markov state :math:`i` in thermodynamic
        state :math:`k`.
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
        r""" The underlying :class:`MarkovStateModelCollection <deeptime.markov.msm.MarkovStateModelCollection>`.
        Contains one Markov state model for each sampled thermodynamic state.

        :getter: The collection of markov state models containing one model for each thermodynamic state.
        :type: MarkovStateModelCollection
        """
        return self._msm_collection

    def compute_sample_weights_log(self, dtrajs, bias_matrices, therm_state=-1):
        r""" Compute the log of the sample weight :math:`\mathrm{log}\;\mu(x)` for all samples :math:`x`.
        If the thermodynamic state index is >= 0, the sample weights for that thermodynamic state will be computed,
        i.e. :math:`\mu^k(x)`. Otherwise, this gives the unbiased sample weights :math:`\mu(x)`.

        Parameters
        ----------
        dtrajs : list(np.ndarray)
            The list of discrete trajectories. `dtrajs[i][n]` contains the Markov state index of the :math:`n`-th sample
            in the :math:`i`-th trajectory. Sample weights are computed for each of the samples in the `dtrajs`.
        bias_matrices : list(np.ndarray)
            The bias energy matrices. `bias_matrices[i][n, k]` contains the bias energy of the :math:`n`-th sample from
            the :math:`i`-th trajectory, evaluated at thermodynamic state :math:`k`, i.e. :math:`b^k(x_{i,n})`.
        therm_state : int, optional
            The index of the thermodynamic state in which the sample weights need to be computed. If `therm_state=-1`,
            the unbiased sample weights are computed.

        Returns
        -------
        sample_weights_log : np.ndarray
            The log of the statistical weight :math:`\mu(x)` of each sample (i.e., the probability distribution over all
            samples: the sum over all sample weights equals one.)

        Notes
        -----
        The statistical distribution is given by

        .. math:: \mu(x) = \left( \sum_k R^k_{i(x)} \mathrm{exp}[f^k_{i(k)}-b^k(x)] \right)^{-1}
        """
        # flatten input data
        dtraj = np.concatenate(dtrajs)
        bias_matrix = np.concatenate(bias_matrices)

        sample_weights = self._compute_sample_weights_log(dtraj, bias_matrix, therm_state)

        # return in the original list shape
        traj_start_stops = np.concatenate(([0], np.cumsum([len(traj) for traj in dtrajs])))
        return [sample_weights[traj_start_stops[i - 1]:traj_start_stops[i]] for i in range(1, len(traj_start_stops))]

    def compute_observable(self, observable_values, dtrajs, bias_matrices, therm_state=-1):
        r""" Compute an observable value.

        Parameters
        ----------
        observable_values : list(np.ndarray)
            The list of observable values. `observable_values[i][n]` contains the observable value for the :math:`n`-th
            sample in the :math:`i`-th trajectory.
        dtrajs : list(np.ndarray)
            The list of discrete trajectories. `dtrajs[i][n]` contains the Markov state index of the :math:`n`-th sample
            in the :math:`i`-th trajectory.
        bias_matrices : list(np.ndarray)
            The bias energy matrices. `bias_matrices[i][n, k]` contains the bias energy of the :math:`n`-th sample from
            the :math:`i`-th trajectory, evaluated at thermodynamic state :math:`k`, :math:`b^k(x_{i,n})`. The bias
            energy matrices should have the same size as `dtrajs` in both the first and second dimension. The third
            dimension is of size `n_therm_state`, i.e. for each sample, the bias energy in every thermodynamic state is
            calculated and stored in the `bias_matrices`.
        therm_state : int, optional, default=-1
            The index of the thermodynamic state in which the observable need to be computed. If `therm_state=-1`, the
            observable is computed for the unbiased (reference) state.

        Returns
        -------
        observable : float
            The expected value of the observable in the selected thermodynamic state `therm_state`.
        """
        # flatten input data
        observable_values = np.concatenate(observable_values)

        sample_weights = self._compute_sample_weights_log(np.concatenate(dtrajs), np.concatenate(bias_matrices),
                                                          therm_state)

        return np.dot(np.exp(sample_weights), observable_values)

    def compute_PMF(self, dtrajs, bias_matrices, bin_indices, n_bins=None, therm_state=-1):
        r""" Compute the potential of mean force over a number of bins.

        Parameters
        ----------
        dtrajs : list(np.ndarray)
            The list of discrete trajectories. `dtrajs[i][n]` contains the Markov state index of the :math:`n`-th sample
            in the :math:`i`-th trajectory.
        bias_matrices : list(np.ndarray)
            The bias energy matrices. `bias_matrices[i][n, k]` contains the bias energy of the :math:`n`-th sample from
            the :math:`i`-th trajectory, evaluated at thermodynamic state :math:`k`, :math:`b^k(x_{i,n})`. The bias
            energy matrices should have the same size as `dtrajs` in both the first and second dimension. The third
            dimension is of size `n_therm_state`, i.e. for each sample, the bias energy in every thermodynamic state is
            calculated and stored in the `bias_matrices`.
        bin_indices : list(np.ndarray)
            The list of bin indices that the samples are binned into. The PMF is calculated as a distribution over these
            bins. `binned_samples[i][n]` contains the bin index for the :math:`n`-th sample in the :math:`i`-th
            trajectory.
        n_bins : int, optional
            The number of bins. If None, n_bins is inferred from the maximum bin index. The PMF array
        therm_state : int, optional, default=-1
            The index of the thermodynamic state in which the PMF need to be computed. If `therm_state=-1`, the PMF is
            computed for the unbiased (reference) state.

        Returns
        -------
        PMF : np.ndarray
            A ndarray of size n_bins containing the estimated PMF from the data. Is n_bins was None, the PMF is of size
            max(bin_indices) + 1.
        """
        # TODO: account for variable bin widths
        sample_weights = self._compute_sample_weights_log(np.concatenate(dtrajs), np.concatenate(bias_matrices),
                                                          therm_state)

        binned_samples = np.concatenate(bin_indices)

        if n_bins is None:
            n_bins = binned_samples.max() + 1
        pmf = np.zeros(n_bins)

        for i in range(len(pmf)):
            indices = np.where(binned_samples == i)
            if len(indices[0]) > 0:
                pmf[i] = -logsumexp(sample_weights[indices])
            else:
                pmf[i] = np.inf
        return pmf

    def compute_log_likelihood(self, dtrajs, bias_matrices):
        r"""The (parameter-dependent part of the) likelihood to observe the given data.

        The definition can be found in :footcite:`wu2016multiensemble`, Equation (9).

        Parameters
        ----------
        dtrajs : list(np.ndarray)
            The list of discrete trajectories. `dtrajs[i][n]` contains the Markov state index of the :math:`n`-th sample
            in the :math:`i`-th trajectory.
        bias_matrices : list(np.ndarray)
            The bias energy matrices. `bias_matrices[i][n, k]` contains the bias energy of the :math:`n`-th sample from
            the :math:`i`-th trajectory, evaluated at thermodynamic state :math:`k`, :math:`b^k(x_{i,n})`. The bias
            energy matrices should have the same size as `dtrajs` in both the first and second dimension. The third
            dimension is of size `n_therm_state`, i.e. for each sample, the bias energy in every thermodynamic state is
            calculated and stored in the `bias_matrices`.

        Returns
        -------
        log_likelihood : float
            The parameter-dependent part of the log-likelihood.


        Notes
        -----
        Parameter-dependent, i.e., the factor

        .. math:: \prod_{x \in X} e^{-b^{k(x)}(x)}

        does not occur in the log-likelihood as it is constant with respect to the parameters, leading to

        .. math:: \log \prod_{k=1}^K \left(\prod_{i,j} (p_{ij}^k)^{c_{ij}^k}\right) \left(\prod_{i} \prod_{x \in X_i^k} \mu(x) e^{f_i^k} \right)
        """
        dtraj = np.concatenate(dtrajs)
        bias_matrix = np.concatenate(bias_matrices)

        transition_counts = transition_counts_from_count_models(self.n_therm_states, self.n_markov_states,
                                                                self._count_models)

        state_counts = state_counts_from_count_models(self.n_therm_states, self.n_markov_states, self._count_models)

        return tram.compute_log_likelihood(dtraj, bias_matrix, self._biased_conf_energies,
                                           self._modified_state_counts_log, self._therm_state_energies, state_counts,
                                           transition_counts, self._transition_matrices)

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

    def _compute_sample_weights_log(self, dtraj, bias_matrix, therm_state=-1):
        sample_weights = np.asarray(tram.compute_sample_weights_log(dtraj, bias_matrix, self._therm_state_energies,
                                                         self._modified_state_counts_log, therm_state))
        # normalize (sum over sample weigths should equal one)
        return sample_weights - logsumexp(sample_weights)
