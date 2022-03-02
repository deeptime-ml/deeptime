import numpy as np

from .decorators import plotting_function
from .platform import handle_progress_bar
from ..base import Observable, BayesianModel, Estimator


def implied_timescales(models, n_its=None):
    r""" Converts a list of models to a :class:`ImpliedTimescales` object.

    .. plot:: examples/plot_implied_timescales.py

    Parameters
    ----------
    models : list
        The input data. Models with and without samples to compute confidences should not be mixed.
    n_its : int or None, optional
        Number of timescales to compute.

    Returns
    -------
    its_data : ImpliedTimescales
        The data object.

    See Also
    --------
    deeptime.plots.plot_implied_timescales
    """
    if not isinstance(models, (list, tuple)):
        models = [models]

    if len(models) == 0:
        raise ValueError("Data cannot be empty.")
    assert all(callable(getattr(model, 'timescales', None)) for model in models), \
        "all models need to have a timescales method"
    assert all(hasattr(model, 'lagtime') for model in models), "all models need a lagtime attribute or property"

    lagtimes = []
    its = []
    its_stats = []

    for model in models:
        is_bayesian = isinstance(model, BayesianModel)
        lagtimes.append(model.lagtime)
        if is_bayesian:
            result = model.timescales(k=n_its)
            its.append(result[0])
            its_stats.append(result[1])
        else:
            its.append(model.timescales(k=n_its))
            its_stats.append(None)
    return ImpliedTimescales(lagtimes, its, its_stats)


class ImpliedTimescales:
    r""" Instances of this class hold a sequence of lagtimes and corresponding process timescales (potentially
    with process timescales of sampled models in a Bayesian setting). Objects can be
    used with :meth:`plot_implied_timescales <deeptime.plots.plot_implied_timescales>`.

    In case models over a range of lagtimes are available, the static method :meth:`from_models` can be used.

    Parameters
    ----------
    lagtimes : iterable of int
        Lagtimes corresponding to processes and their timescales.
    its : iterable of ndarray
        The timescales for each process, shape (n_lagtimes, n_processes).
    its_stats : list of list of ndarray, optional, default=None
        Sampled timescales of shape (n_lagtimes, n_processes, n_samples).

    See Also
    --------
    deeptime.plots.plot_implied_timescales
    """

    def __init__(self, lagtimes, its, its_stats=None):
        self._lagtimes = np.asarray(lagtimes, dtype=int)
        assert len(its) == self.n_lagtimes, f"The length of its should correspond to the number of " \
                                            f"lagtimes ({self.n_lagtimes}), got {len(its)} instead."
        self._max_n_processes = max(len(x) for x in its)
        self._max_n_samples = 0 if its_stats is None else max(len(x) if x is not None else 0 for x in its_stats)
        self._its = np.full((self.n_lagtimes, self._max_n_processes), fill_value=np.nan)
        assert self._its.ndim == 2 and self._its.shape[0] == self.n_lagtimes, \
            "its should be of shape (lagtimes, processes)."
        for i, processes in enumerate(its):
            self._its[i, :len(processes)] = processes

        if self.has_samples:
            assert len(its_stats) == self.n_lagtimes, f"The length of its stats should correspond to the number of " \
                                                      f"lagtimes ({self.n_lagtimes}), got {len(its_stats)} instead."
            self._its_stats = np.full((self.n_lagtimes, self.max_n_processes, self.max_n_samples), fill_value=np.nan)
            for lag_ix in range(len(its_stats)):

                samples = its_stats[lag_ix]
                if samples is not None:
                    for sample_ix in range(len(samples)):
                        arr = np.asarray(its_stats[lag_ix][sample_ix])
                        n = min(len(arr), self.max_n_processes)
                        self._its_stats[lag_ix, :n, sample_ix] = arr[:n]
            if not (self._its_stats.ndim == 3 and self._its_stats.shape[0] == self.n_lagtimes and
                    self._its_stats.shape[1] == self.max_n_processes):
                raise ValueError(f"its_stats should be of shape (lagtimes={self.n_lagtimes}, "
                                 f"processes={self.max_n_processes}, samples={self.max_n_samples}) but was "
                                 f"{self._its_stats.shape}")
        else:
            self._its_stats = None
        ix = np.argsort(self.lagtimes)
        self._lagtimes = self._lagtimes[ix]
        self._its = self._its[ix]
        self._its_stats = None if self._its_stats is None else self._its_stats[ix]

    @property
    def lagtimes(self) -> np.ndarray:
        r""" Yields the lagtimes corresponding to an instance of this class. """
        return self._lagtimes

    @property
    def n_lagtimes(self) -> int:
        r""" Number of lagtimes. """
        return len(self.lagtimes)

    @property
    def max_n_processes(self) -> int:
        r""" Maximum number of processes. """
        return self._max_n_processes

    @property
    def max_n_samples(self) -> int:
        r""" Maximum number of samples. """
        return self._max_n_samples

    @property
    def has_samples(self) -> bool:
        r""" Whether the data contains samples. """
        return self.max_n_samples > 0

    def timescales_for_process(self, process_index: int) -> np.ndarray:
        r""" Yields maximum-likelihood timescales for a particular process.

        Parameters
        ----------
        process_index : int
            The process.

        Returns
        -------
        timescales : ndarray (lagtimes,)
            The timescales for the particular process. Might contain NaN.
        """
        assert process_index < self.max_n_processes, \
            f"The process ({process_index}) should be contained in data ({self.max_n_processes})."
        return self._its[:, process_index]

    def samples_for_process(self, process_index: int) -> np.ndarray:
        r"""Yields timescales samples for a particular process.

        Parameters
        ----------
        process_index : int
            The process.

        Returns
        -------
        timescales_samples : ndarray(lagtimes, max_n_samples)
            The sampled timescales for a particular process. Might contain NaN.
        """
        assert self.has_samples, "This timescales data object contains no samples."
        assert process_index < self.max_n_processes, "The process should be contained in data."
        return self._its_stats[:, process_index]

    def n_samples(self, lagtime_index: int, process_index: int) -> int:
        r""" Yields the number of samples for a particular lagtime and a particular process.

        Parameters
        ----------
        lagtime_index : int
            The lagtime index corresponding to :attr:`lagtimes`.
        process_index : int
            The process index.

        Returns
        -------
        n_samples : int
            The number of samples.
        """
        data = self.samples_for_process(process_index)[lagtime_index]
        return np.count_nonzero(~np.isnan(data))

    @plotting_function()
    def plot(self, *args, **kw):
        r""" Dispatches to :meth:`plot_implied_timescales`. """
        from deeptime.plots import plot_implied_timescales
        plot_implied_timescales(self, *args, **kw)


def ck_test(models, observable: Observable, test_model=None, include_lag0=True, err_est=False, progress=None):
    r""" Performs a Chapman-Kolmogorov test: Under the assumption of Markovian dynamics some transfer operators
    such as the transition matrix of a Markov state model or a Koopman model possess the so-called Chapman-Kolmogorov
    property. It states that for lagtimes :math:`\tau_1, \tau_2` the transfer operator behaves like
    :math:`\mathcal{T}_{\tau_1 + \tau_2} = \mathcal{T}_{\tau_1}\mathcal{T}_{\tau_2}`.

    This method performs a test to verify the above identity by evaluating an observable :math:`\rho` *twice*
    on a range of different lagtimes. The observable evaluated on each model (estimated at a different lagtime)
    is compared against the observable evaluated on the test model when it is propagated by the Chapman-Kolmogorov
    equation to the corresponding lagtime:

    .. math::
        \mathcal{T}^\mathrm{test}_{k \tau}\rho \overset{!}{=} \mathcal{T}^{\mathrm{model}}_{\tilde{\tau}}

    such that :math:`k\tau = \tilde{\tau}`.

    .. plot:: examples/plot_ck_test.py

    Parameters
    ----------
    models : list of models
        List of models estimated at different lagtimes.
    observable : Observable
        An observable. The choice mainly depends on what kind of models are compared. Observables are required
        to be callable with a model instance.
    test_model : model, optional, default=None
        The test model. Per default this uses the model with the smallest lagtime of the models list.
    include_lag0 : bool, optional, default=True
        Whether to include :math:`\tau = 0` into the test.
    err_est : bool, optional, default=False
        Whether to compute the observable on Bayesian samples of the estimated models.
    progress : ProgressBar, optional, default=None
        An optional progress bar. Tested with tqdm.

    Returns
    -------
    ck_test : ChapmanKolmogorovTest
        The results. Can be used with `plot_ck_test <deeptime.plots.plot_ck_test>`.

    See Also
    --------
    deeptime.plots.plot_ck_test
    """
    assert all(hasattr(m, 'lagtime') for m in models) and (test_model is None or hasattr(test_model, 'lagtime')), \
        "All models and the test model need to have a lagtime property."
    progress = handle_progress_bar(progress)
    models = sorted(models, key=lambda x: x.lagtime)
    if test_model is None:
        test_model = models[0]

    is_bayesian = isinstance(test_model, BayesianModel)
    assert all(isinstance(m, BayesianModel) == is_bayesian for m in models), \
        "All models must either have samples or none at all."
    lagtimes = [0] if include_lag0 else []
    lagtimes += [m.lagtime for m in models]

    predictions = []
    predictions_samples = []
    reference_lagtime = test_model.lagtime
    for lagtime in progress(lagtimes):
        predictions.append(observable(test_model, mlag=lagtime / reference_lagtime))
        if is_bayesian:
            obs = []
            for sample in test_model.samples:
                obs.append(observable(sample, mlag=lagtime / reference_lagtime))
            predictions_samples.append(obs)

    if include_lag0:
        models = [None] + models
    estimates = []
    estimates_samples = []
    for model in progress(models):
        estimates.append(observable(model, mlag=1))
        if is_bayesian and err_est:
            if model is not None:
                obs = []
                for sample in model.samples:
                    obs.append(observable(sample, mlag=1))
                else:
                    obs.append(observable(model, mlag=0))
                estimates_samples.append(obs)
            else:
                estimates_samples.append([observable(model, mlag=0)])

    return ChapmanKolmogorovTest(lagtimes, predictions, predictions_samples, estimates, estimates_samples,
                                 observable)


class ChapmanKolmogorovTest:
    r""" Test results of the Chapman-Kolmogorov test. See :meth:`ck_test`. """

    def __init__(self, lagtimes, predictions, predictions_samples, estimates, estimates_samples, observable):
        self._lagtimes = np.array(lagtimes)
        self._predictions = np.asarray(predictions)
        self._predictions_samples = predictions_samples
        self._estimates = np.asarray(estimates)
        self._estimates_samples = estimates_samples
        self._observable = observable

    @property
    def n_components(self):
        return len(self.estimates[0]) if self.estimates is not None and len(self.estimates) > 0 else 0

    @property
    def lagtimes(self):
        r""" Lagtimes at which estimations and predictions were performed. """
        return self._lagtimes

    @property
    def estimates(self):
        """ Estimates at different lagtimes.

        :getter: each row contains the n observables computed at one of the T lagtimes.
        :type: ndarray(T, n, n)
        """
        return self._estimates

    @property
    def estimates_samples(self):
        r""" Returns the sampled estimates, i.e., a list of arrays as described in :attr:`estimates`. Can be `None`. """
        return self._estimates_samples

    @property
    def predictions(self):
        """ Returns tested model predictions at different lagtimes

        Returns
        -------
        Y : ndarray(T, n)
            each row contains the n observables predicted at one of the T lag
            times by the tested model.

        """
        return self._predictions

    @property
    def predictions_samples(self):
        """ Returns the confidence intervals of the estimates at different
        lagtimes (if available)

        If not available, returns None.

        Returns
        -------
        L : ndarray(T, n)
            each row contains the lower confidence bound of n observables
            computed at one of the T lag times.

        R : ndarray(T, n)
            each row contains the upper confidence bound of n observables
            computed at one of the T lag times.

        """
        return self._predictions_samples

    @property
    def has_errors(self):
        r""" Whether the prediction contains samples. """
        return self.predictions_samples is not None and len(self.predictions_samples) > 0

    @property
    def err_est(self):
        r""" Whether the estimated models contain samples """
        return self.estimates_samples is not None and len(self.estimates_samples) > 0


class DeprecatedCKValidator(Estimator):

    def __init__(self, estimator, fit_for_lag, mlags, observable, test_model):
        super().__init__()
        self.estimator = estimator
        self.mlags = mlags
        self.fit_for_lag = fit_for_lag
        self.observable = observable
        self.test_model = test_model

    def fit(self, data, **kwargs):
        if hasattr(self.test_model, 'prior'):
            test_lag = self.test_model.prior.lagtime
        else:
            test_lag = self.test_model.lagtime
        models = []
        for factor in range(1, self.mlags):
            lagtime = factor * test_lag
            models.append(self.fit_for_lag(data, lagtime))
        self._model = ck_test(models, observable=self.observable, test_model=self.test_model)
        return self
