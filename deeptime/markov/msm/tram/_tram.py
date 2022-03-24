import warnings
import numpy as np
from typing import Optional
from sklearn.exceptions import ConvergenceWarning

from deeptime.markov._base import _MSMBaseEstimator
from deeptime.util import callbacks
from ._tram_bindings import tram
from ._tram_model import TRAMModel
from ._tram_dataset import TRAMDataset


def _unpack_input_tuple(data):
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


def _get_dataset_from_input(data):
    if isinstance(data, tuple):
        dtrajs, bias_matrices, ttrajs = _unpack_input_tuple(data)
        return TRAMDataset(dtrajs=dtrajs, bias_matrices=bias_matrices, ttrajs=ttrajs)
    if isinstance(data, TRAMDataset):
        return data

    raise ValueError("Input data invalid. Input data type should be of tuple or TRAMDataset.")


class TRAM(_MSMBaseEstimator):
    r"""Transition(-based) Reweighting Analysis Method.

    TRAM is described in :footcite:`wu2016multiensemble`. The parameters in this code correspond to variables in the
    TRAM paper in the following way:

    * `biased_conf_energies[k][i]` corresponds to :math:`f_i^k`, the free energy for  Markov state i in
      thermodynamic state k.
    * `lagrangian_mult_log[k][i]` corresponds to :math:`log\;v_i^k`, the logarithm of the lagrangian multiplier of
      Markov state i in thermodynamic state k.
    * `modified_state_counts_log[k][i]` corresponds to :math:`log\;R_i^k`, the logarithm of :math:`R_i^k`, the modified
      state counts for Markov state i in thermodynamic state k.

    Parameters
    ----------
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

    maxiter : int, optional, default=1000
        The maximum number of self-consistent iterations before the estimator exits unsuccessfully.
    maxerr : float, optional, default=1E-8
        Convergence criterion based on the maximal free energy change in a self-consistent
        iteration step.
    init_strategy : str, optional, default='MBAR'
        Strategy of initialization of the free energies, lagrangian multipliers and state counts. Possible choices:
        "MBAR" or None.

         * "MBAR" : Initialize free energies using MBAR :footcite:`shirts2008statistically`. MBAR iterations are
           executed before TRAM, to initialize the free energies with the MBAR estimate, which is a good initial
           estimate for the TRAM free energies and will significantly speed up the convergence of TRAM. Lagrangian
           multipliers are initialized to :math:`v_i^k = log( 1/2 * \sum_j (c_ij^k + c_ji^k))` and modified state counts
           are zero-initialized.
         * None : Free energies and modified state counts are zero-initialized. Lagrangian multipliers are initialized
           to :math:`v_i^k = log( 1/2 * \sum_j (c_ij^k + c_ji^k))`

    init_maxiter : int, optional, default=1000
        The maximum number of iterations for parameter initialization, e.g. MBAR iterations. These initialization
        iterations are executed before TRAM, to initialize the parameters with the chosen `init_strategy`.
    init_maxerr : float, optional, default = 1eE-8
        Convergence criterion for the initialization routine, based on the maximum energy change after one iteration
        step.
    track_log_likelihoods : bool, optional, default=False
        If `True`, the log-likelihood is stored every callback_interval steps. For calculation of the log-likelihood the
        transition matrix needs to be constructed, which will slow down estimation. By default, log-likelihoods are
        not computed.
    callback_interval : int, optional, default=0
        Every `callback_interval` iteration steps, the callback function is called and error increments are stored. If
        `track_log_likelihoods=true`, the log-likelihood are also stored. If `callback_interval=0`, no call to the
        callback function is done.
    progress : object
        Progress bar object that `TRAM` will call to indicate progress to the user. Tested for a tqdm progress bar.
        The interface is checked
        via :meth:`supports_progress_interface <deeptime.util.callbacks.supports_progress_interface>`.

    See also
    --------
    :class:`TRAMDataset <deeptime.markov.msm.TRAMDataset>`, :class:`TRAMModel <deeptime.markov.msm.TRAMModel>`

    References
    ----------
    .. footbibliography::
    """

    def __init__(
            self, lagtime=1, count_mode='sliding',
            maxiter=1000, maxerr: float = 1e-8,
            init_strategy='MBAR',
            init_maxiter=1000, init_maxerr=1e-8,
            track_log_likelihoods=False, callback_interval=1,
            progress=None):

        super(TRAM, self).__init__()

        self.lagtime = lagtime
        self.count_mode = count_mode
        self._tram_estimator = None
        self.maxiter = maxiter
        self.maxerr = maxerr
        self.init_strategy = init_strategy
        self.init_maxiter = init_maxiter
        self.init_maxerr = init_maxerr
        self.track_log_likelihoods = track_log_likelihoods
        self.callback_interval = callback_interval
        self.progress = progress
        self._largest_connected_set = None
        self.log_likelihoods = []
        self.energy_increments = []

    init_strategy_options = ["MBAR", None]

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
        data: TRAMDataset or tuple
            If data is supplied in form of a data tuple, a TRAMDataset is constructed from the data tuple inside
            `fit()`. The data tuple is of shape `(dtrajs, bias_matrices)` or `(dtrajs, bias_matrices, ttrajs)`, with the
            inner values given by:

            * dtrajs: array-like(ndarray(n)), int
               The discrete trajectories in the form of a list or array of numpy arrays. `dtrajs[i]` contains one
               trajectory. `dtrajs[i][n]` equals the Markov state index that the :math:`n`-th sample from the
               :math:`i`-th trajectory was binned into. Each of the `dtrajs` can be of variable length.
            * bias_matrices: ndarray-like(ndarray(n,m)), float
               The bias energy matrices. `bias_matrices[i][n, k]` equals the bias energy of the :math:`n`-th sample from
               the :math:`i`-th trajectory, evaluated at thermodynamic state :math:`k`, :math:`b^k(x_{i,n})`. The bias
               energy matrices should have the same size as `dtrajs` in both the first and second dimensions. The third
               dimension is of size `n_therm_states`, i.e. for each sample, the bias energy in every thermodynamic state
               is calculated and stored in the `bias_matrices`.
            * ttrajs: array-like(ndarray(n)), int, optional
               `ttrajs[i]` indicates for each sample in the :math:`i`-th trajectory what thermodynamic state that sample
               was sampled at. If `ttrajs = None`, we assume no replica exchange was done. In this case we assume each
               trajectory  corresponds to a unique thermodynamic state, and `n_therm_states` equals the size of
               `dtrajs`.
        model : TRAMModel, optional, default=None
            If a TRAMModel is given, the parameters from the TRAMModel are loaded into the estimator, and estimation
            continues from the loaded parameters as a starting point. Input data may differ from the input data used to
            estimate the input model, but the input data should lie within bounds of the number of thermodynamic states
            and Markov states given by the model, meaning the highest occurring state indices in `ttrajs` and `dtrajs`
            may be `model.n_therm_states - 1` and `model.n_markov_states - 1` respectively,
            If no model is given, estimation starts from zero-initialized arrays for the free energies and modified
            state counts. The lagrangian multipliers are initialized with values
            :math:`v_i^{k, 0} = \mathrm{log} (c_{ij}^k + c_{ji}^k)/2`

        See also
        --------
        :class:`TRAMDataset <deeptime.markov.msm.tram.TRAMDataset>`
        """
        dataset = _get_dataset_from_input(data)

        if model is not None:
            # check whether the data lies within state bounds of the model
            dataset.check_against_model(model)

        # only construct estimator if it hasn't been loaded from the model yet
        self._tram_estimator = self._make_tram_estimator(model, dataset)

        self._run_estimation(dataset.tram_input)
        self._model = TRAMModel(count_models=dataset.count_models,
                                transition_matrices=self._tram_estimator.transition_matrices,
                                biased_conf_energies=self._tram_estimator.biased_conf_energies,
                                lagrangian_mult_log=self._tram_estimator.lagrangian_mult_log,
                                modified_state_counts_log=self._tram_estimator.modified_state_counts_log,
                                therm_state_energies=self._tram_estimator.therm_state_energies,
                                markov_state_energies=self._tram_estimator.markov_state_energies)

        return self

    def _make_tram_estimator(self, model, dataset):
        r""" Construct the underlying c++ TRAM estimator. If a model is given, the estimator is initialized with the
        model parameters. Otherwise, the free energies are initialized with the chosen initialization strategy.
        """
        if self.init_strategy not in TRAM.init_strategy_options:
            raise ValueError(
                f"Initialization strategy unsupported. init_strategy must be one of {TRAM.init_strategy_options}.")

        if model is not None:
            return tram.TRAM(model.biased_conf_energies, model.lagrangian_mult_log, model.modified_state_counts_log)
        else:
            if self.init_strategy == "MBAR":
                # initialize free energies using MBAR.
                with callbacks.IterationErrorProgressCallback(self.progress, "Initializing free energies using MBAR",
                                                self.init_maxiter) as callback:
                    free_energies = tram.initialize_free_energies_mbar(np.concatenate(dataset.bias_matrices),
                                                                       dataset.state_counts.sum(axis=1),
                                                                       self.init_maxiter, self.init_maxerr,
                                                                       self.callback_interval, callback)

                # copy free energies along the markoc state axis to get initial biased_conf_energies
                biased_conf_energies = np.repeat(free_energies[:, None], dataset.n_markov_states, axis=1)
            else:
                biased_conf_energies = np.zeros((dataset.n_therm_states, dataset.n_markov_states))

        lagrangian_mult_log = tram.initialize_lagrangians(dataset.transition_counts)
        modified_state_counts = np.zeros_like(lagrangian_mult_log)  # intialize this as the dataset state counts???
        return tram.TRAM(biased_conf_energies, lagrangian_mult_log, modified_state_counts)

    def _run_estimation(self, tram_input):
        """ Estimate the free energies using self-consistent iteration as described in the TRAM paper. """
        with TRAMCallback(self.progress, self.maxiter, self.log_likelihoods, self.energy_increments) as callback:
            self._tram_estimator.estimate(tram_input, self.maxiter, self.maxerr,
                                          track_log_likelihoods=self.track_log_likelihoods,
                                          callback_interval=self.callback_interval,
                                          callback=callback)

            if callback.last_increment > self.maxerr:
                warnings.warn(
                    f"TRAM did not converge after {self.maxiter} iteration(s). "
                    f"Last increment: {callback.last_increment}", ConvergenceWarning)


class TRAMCallback(callbacks.IterationErrorProgressCallback):
    """Callback for the TRAM estimate process. Increments a progress bar and saves iteration increments in the free
    energies and log-likelihoods to a list.

    Parameters
    ----------
    log_likelihoods_list : list, optional
        A list to append the log-likelihoods to that are passed to the callback.__call__() method.
    total : int
        Maximum number of callbacks.
    increments : list, optional
        A list to append the increments to that are passed to the callback.__call__() method.
    """

    def __init__(self, progress, total, log_likelihoods_list=None, increments=None):
        super().__init__(progress, total=total, description="Running TRAM estimate")
        self.log_likelihoods = log_likelihoods_list
        self.increments = increments
        self.last_increment = 0

    def __call__(self, inc, error, log_likelihood=0):
        """Call the callback. Increment a progress bar (if available) and store convergence information.

        Parameters
        ----------
        n_iterations : int
            Number of iterations to increment the progress bar with
        increment : float
            The increment in the free energies after the last iteration.
        log_likelihood : float
            The current log-likelihood, or 0. when the tram estimator is not configured to calculate log-likelihoods.
        """
        super().__call__(inc, error=error)

        if self.log_likelihoods is not None:
            self.log_likelihoods.append(log_likelihood)

        if self.increments is not None:
            self.increments.append(error)

        self.last_increment = error
