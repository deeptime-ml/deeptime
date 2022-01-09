import warnings
from typing import Optional
from sklearn.exceptions import ConvergenceWarning

from deeptime.markov._base import _MSMBaseEstimator
from deeptime.util import callbacks
from ._tram_bindings import tram
from ._tram_model import TRAMModel
from ._tram_dataset import TRAMDataset

import numpy as np


def unpack_input_tuple(data):
    """Get input from the data tuple. Data is of variable size.

    Parameters
    ----------
    data : tuple(2) or tuple(3)
        data[0] contains the dtrajs. data[1] the bias matrix, and data[2] may or may not contain the ttrajs.

    Returns
    -------
    dtrajs : array-like(ndarray(n)), int32
        the first element from the data tuple.
    bias_matrices : array-like(ndarray(n,m)), float64
        the second element from the data tuple.
    ttrajs : array-like(ndarray(n)), int32, or None
        the third element from the data tuple, if present.
    """
    dtrajs, bias_matrices, ttrajs = data[0], data[1], data[2:]
    if ttrajs is not None and len(ttrajs) > 0:
        if len(ttrajs) > 1:
            raise ValueError("Unexpected number of arguments in data tuple.")
        ttrajs = ttrajs[0]

    return dtrajs, bias_matrices, ttrajs


class TRAM(_MSMBaseEstimator):
    r"""Transition(-based) Reweighting Analysis Method.
    TRAM is described in :footcite:`wu2016multiensemble`.

    Parameters
    ----------
    model : TRAMModel, or None
        If TRAM is initialized with a TRAMModel, the parameters from the TRAMModel are loaded into the estimator, and
        estimation continues from the loaded parameters as a starting point. Input data may differ from the input data
        used to estimate the input model, but the input data should lie within bounds of the number of thermodynamic
        states and Markov states given by the model.
        If no model is given, estimation starts from zero-initialized arrays for the free energies and modified state
        counts. The lagrangian multipliers are initialized with values
            :math:`v_i^{k, 0} = \mathrm{log} (c_{ij}^k + c_{ji}^k)/2`
    lagtime : int, default=1
        Integer lag time at which transitions are counted.
    count_mode : str, default="sliding"
        One of "sample", "sliding", "sliding-effective", and "effective".

        * "sample" strides the trajectory with lagtime :math:`\tau` and uses the strided counts as transitions.
        * "sliding" uses a sliding window approach, yielding counts that are statistically correlated and too
          large by a factor of :math:`\tau`; in uncertainty estimation this yields wrong uncertainties.
        * "sliding-effective" takes "sliding" and divides it by :math:`\tau`, which can be shown to provide a
          likelihood that is the geometrical average over shifted subsamples of the trajectory,
          :math:`(s_1,\:s_{tau+1},\:...),\:(s_2,\:t_{tau+2},\:...),` etc. This geometrical average converges to
          the correct likelihood in the statistical limit :footcite:`trendelkamp2015estimation`.
        * "effective" uses an estimate of the transition counts that are statistically uncorrelated.
          Recommended when estimating Bayesian MSMs.
    connectivity : str, optional, default="post_hoc_RE"
        One of "post_hoc_RE", "BAR_variance", or "summed_count_matrix".
        Defines what should be considered a connected set in the joint (product) space
        of conformations and thermodynamic ensembles.

        * "post_hoc_RE" : It is required that every state in the connected set can be reached by following a
          pathway of reversible transitions or jumping between overlapping thermodynamic states while staying in
          the same Markov state. A reversible transition between two Markov states (within the same thermodynamic
          state k) is a pair of Markov states that belong to the same strongly connected component of the count
          matrix (from thermodynamic state k). Two thermodynamic states k and l are defined to overlap at Markov
          state n if a replica exchange simulation :footcite:`hukushima1996exchange` restricted to state n would
          show at least one transition from k to l or one transition from from l to k.
          The expected number of replica exchanges is estimated from
          the simulation data. The minimal number required of replica exchanges per Markov state can be increased by
          decreasing `connectivity_factor`.
        * "BAR_variance" : like 'post_hoc_RE' but with a different condition to define the thermodynamic overlap
          based on the variance of the BAR estimator :footcite:`shirts2008statistically`.
          Two thermodynamic states k and l are defined to overlap
          at Markov state n if the variance of the free energy difference Delta :math:`f_{kl}` computed with BAR (and
          restricted to conformations form Markov state n) is less or equal than one. The minimally required variance
          can be controlled with `connectivity_factor`.
        * "summed_count_matrix" : all thermodynamic states are assumed to overlap. The connected set is then
          computed by summing the count matrices over all thermodynamic states and taking its largest strongly
          connected set. Not recommended!
    maxiter : int, optional, default=10000
        The maximum number of self-consistent iterations before the estimator exits unsuccessfully.
    maxerr : float, optional, default=1E-15
        Convergence criterion based on the maximal free energy change in a self-consistent
        iteration step.
    track_log_likelihoods : bool, optional, default=False
        If True, the log-likelihood is stored every callback_interval steps. For calculation of the log-likelihood the
        transition matrix needs to be constructed, which will slow down estimation. By default, log-likelihoods are
        not computed.
    callback_interval : int, optional, default=0
        Every callback_interval iteration steps, the callback function is calles and error increments are stored. If
        track_log_likelihoods=true, the log-likelihood are also stored. If 0, no call to the callback function is done.
    connectivity_factor : float, optional, default=1.0
        Only needed if connectivity="post_hoc_RE" or "BAR_variance". Values greater than 1.0 weaken the connectivity
        conditions. For 'post_hoc_RE' this multiplies the number of hypothetically observed transitions. For
        'BAR_variance' this scales the threshold for the minimal allowed variance of free energy differences.
    progress_bar : object
        Progress bar object that TRAM will call to indicate progress to the user.
        Tested for a tqdm progress bar. Should implement update() and close() and have .total and .desc properties.

    References
    ----------
    .. footbibliography::
    """

    def __init__(
            self, lagtime=1, count_mode='sliding',
            connectivity='summed_count_matrix',
            maxiter=10000, maxerr: float = 1e-8, track_log_likelihoods=False,
            callback_interval=0,
            connectivity_factor: float = 1.0,
            progress_bar=None):

        super(TRAM, self).__init__()

        self.lagtime = lagtime
        self.count_mode = count_mode
        self._tram_estimator = None

        if connectivity not in TRAM.connectivity_options:
            raise ValueError(f"Connectivity type unsupported. Connectivity must be one of {TRAM.connectivity_options}.")
        self.connectivity = connectivity
        self.connectivity_factor = connectivity_factor
        self.maxiter = maxiter
        self.maxerr = maxerr
        self.track_log_likelihoods = track_log_likelihoods
        self.callback_interval = callback_interval
        self.progress_bar = progress_bar
        self._largest_connected_set = None
        self.log_likelihoods = []
        self.increments = []

    #: All possible connectivity modes
    connectivity_options = ["post_hoc_RE", "BAR_variance", "summed_count_matrix", None]

    @property
    def compute_log_likelihood(self) -> Optional[float]:
        r"""The parameter-dependent part of the TRAM likelihood.

        The definition can be found in :footcite:`wu2016multiensemble`, Equation (9).

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
        if self._tram_estimator is not None:
            return self._tram_estimator.compute_log_likelihood()

    def fetch_model(self) -> Optional[TRAMModel]:
        r"""Yields the most recent :class:`MarkovStateModelCollection` that was estimated.
        Can be None if fit was not called.

        Returns
        -------
        model : MarkovStateModelCollection or None
            The most recent markov state model or None.
        """
        return self._model

    def fit(self, data, model=None, *args, **kw):
        r"""Fit a MarkovStateModelCollection to the given input data, using TRAM.
        For each thermodynamic state, there is one Markov Model in the collection at the respective thermodynamic state
        index.

        Parameters
        ----------
        data: tuple consisting of (dtrajs, bias_matrices) or (dtrajs, bias_matrices, ttrajs).
            * dtrajs: array-like(ndarray(n)), int32
              The discrete trajectories in the form of a list or array of numpy arrays. dtrajs[i] contains one trajectory.
              dtrajs[i][n] contains the Markov state index that the n-th sample from the i-th trajectory was binned
              into. Each of the dtrajs can be of variable length.
            * bias_matrices: ndarray-like(ndarray), float64
              The bias energy matrices. bias_matrices[i, n, l] contains the bias energy of the n-th sample from the i-th
              trajectory, evaluated at thermodynamic state k. The bias energy matrices should have the same size as
              dtrajs in both the 0-th and 1-st dimension. The seconds dimension of of size n_therm_state, i.e. for each
              sample, the bias energy in every thermodynamic state is calculated and stored in the bias_matrices.
            * ttrajs: array-like(ndarray], int32, optional
              ttrajs[i] indicates for each sample in the i-th trajectory what thermodynamic state that sample was
              sampled at. If ttrajs is None, we assume no replica exchange was done. In this case we assume each
              trajectory  corresponds to a unique thermodynamic state, and n_therm_states equals the size of dtrajs.
        """

        if isinstance(data, tuple):
            dtrajs, bias_matrices, ttrajs = unpack_input_tuple(data)
            dataset = TRAMDataset(dtrajs=dtrajs, bias_matrices=bias_matrices, ttrajs=ttrajs)
        if isinstance(data, TRAMDataset):
            dataset = data

        if model is not None:
            self._model = model
            dataset.n_markov_states = self._model.n_markov_states
            dataset.n_therm_states = self._model.n_therm_states

        # only construct estimator if it hasn't been loaded from the model yet
        self._tram_estimator = self._make_tram_estimator(dataset.n_markov_states, dataset.n_therm_states)

        self._run_estimation(dataset.tram_input)
        self._model = TRAMModel(count_models=dataset.count_models,
                                transition_matrices=self._tram_estimator.transition_matrices,
                                biased_conf_energies=self._tram_estimator.biased_conf_energies,
                                lagrangian_mult_log=self._tram_estimator.lagrangian_mult_log,
                                modified_state_counts_log=self._tram_estimator.modified_state_counts_log,
                                therm_state_energies=self._tram_estimator.therm_state_energies,
                                markov_state_energies=self._tram_estimator.markov_state_energies)

        return self

    def _run_estimation(self, tram_input):
        """ Estimate the free energies using self-consistent iteration as described in the TRAM paper.
        """
        with TRAMCallback(self.progress_bar, self.maxiter, self.log_likelihoods, self.increments,
                          self.callback_interval > 0) as callback:
            self._tram_estimator.estimate(tram_input, self.maxiter, self.maxerr,
                                          track_log_likelihoods=self.track_log_likelihoods,
                                          callback_interval=self.callback_interval, callback=callback)

            if callback.last_increment > self.maxerr:
                warnings.warn(
                    f"TRAM did not converge after {self.maxiter} iteration. Last increment: {callback.last_increment}",
                    ConvergenceWarning)

    def _make_tram_estimator(self, n_therm_states, n_markov_states):
        if self._model is None:
            return tram.TRAM(n_therm_states, n_markov_states)
        else:
            return tram.TRAM(self._model.biased_conf_energies, self._model.lagrangian_mult_log,
                                         self._model.modified_state_counts_log)


class TRAMCallback(callbacks.Callback):
    """Callback for the TRAM estimate process. Increments a progress bar and optionally saves iteration increments and
    log likelihoods to a list."""

    def __init__(self, progress_bar, n_iter, log_likelihoods_list=None, increments=None, store_convergence_info=False):
        super().__init__(progress_bar, n_iter, "Running TRAM estimate")
        self.log_likelihoods = log_likelihoods_list
        self.increments = increments
        self.store_convergence_info = store_convergence_info
        self.last_increment = 0

    def __call__(self, increment, log_likelihood):
        super().__call__()

        if self.store_convergence_info:
            if self.log_likelihoods is not None:
                self.log_likelihoods.append(log_likelihood)
            if self.increments is not None:
                self.increments.append(increment)

        self.last_increment = increment
