from typing import Optional

import numpy as np

from deeptime.util import confidence_interval
from deeptime.util.decorators import plotting_function


class ImpliedTimescalesData:
    r""" Instances of this class hold a sequence of lagtimes and corresponding process timescales (potentially
    with process timescales of sampled models in a Bayesian setting). Objects can be
    used with :meth:`plot_implied_timescales`.

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
    plot_implied_timescales
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

    @staticmethod
    def from_models(models, n_its=None):
        r""" Converts a list of models to a :class:`ImpliedTimescalesData` object.

        Parameters
        ----------
        models : list
            The input data. Models with and without samples to compute confidences should not be mixed.
        n_its : int or None, optional
            Number of timescales to compute.

        Returns
        -------
        its_data : ImpliedTimescalesData
            The data object.
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
            is_bayesian = hasattr(model, 'prior') and hasattr(model, 'samples')
            lagtimes.append(model.lagtime)
            if is_bayesian:
                result = model.timescales(k=n_its)
                its.append(result[0])
                its_stats.append(result[1])
            else:
                its.append(model.timescales(k=n_its))
                its_stats.append(None)
        return ImpliedTimescalesData(lagtimes, its, its_stats)


@plotting_function
def plot_implied_timescales(ax, data: ImpliedTimescalesData, n_its: Optional[int] = None, process: Optional[int] = None,
                            show_mle: bool = True, show_samples: bool = True, show_sample_mean: bool = True,
                            show_sample_confidence: bool = True, show_cutoff: bool = True,
                            sample_confidence: float = .95,
                            colors=None, **kwargs):
    r"""Creates an implied timescales plot inside exising matplotlib axes.

    .. plot:: examples/plot_implied_timescales.py

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The matplotlib axes to use for plotting.
    data : ImpliedTimescalesData
        A timescales data container object, can be obtained, e.g., via :meth:`ImpliedTimescalesData.from_models`.
    n_its : int, optional, default=None
        Maximum number of timescales to plot.
    process : int, optional, default=None
        A particular process to plot. This is mutually exclusive with n_its.
    show_mle : bool, default=True
        Whether to show the timescale of the maximum-likelihood estimate.
    show_samples : bool, default=True
        Whether to show sample means and/or confidences.
    show_sample_mean : bool, default=True
        Whether to show the sample mean. Only has an effect if show_samples is True and there are samples in the data.
    show_sample_confidence : bool, default=True
        Whether to show the sample confidence. Only has an effect if show_samples is True and there are samples
        in the data.
    show_cutoff : bool, default=True
        Whether to show the model resolution cutoff as grey filled area.
    sample_confidence : float, default=0.95
        The confidence to plot. The default amounts to a shaded area containing 95% of the sampled values.
    colors : list of colors, optional, default=None
        The colors that should be used for timescales. By default uses the matplotlib default colors as per
        rc-config value "axes.prop_cycle".
    **kwargs
        Keyword arguments which are forwarded into the matplotlib plotting function for timescales.

    See Also
    --------
    ImpliedTimescalesData
    """

    if n_its is not None and process is not None:
        raise ValueError("n_its and process are mutually exclusive.")
    if process is not None and process >= data.max_n_processes:
        raise ValueError(f"Requested process {process} when only {data.max_n_processes} are available.")

    if process is None and n_its is None:
        n_its = data.max_n_processes
    it_indices = [process] if process is not None else np.arange(n_its)
    if colors is None:
        from matplotlib import rcParams
        colors = rcParams['axes.prop_cycle'].by_key()['color']
    for it_index in it_indices:
        color = colors[it_index % len(colors)]
        if show_mle:
            ax.plot(data.lagtimes, data.timescales_for_process(it_index), color=color, **kwargs)
        if data.has_samples and show_samples:
            its_samples = data.samples_for_process(it_index)
            if show_sample_mean:
                sample_mean = np.nanmean(its_samples, axis=1)
                ax.plot(data.lagtimes, sample_mean, marker='o', linestyle='dashed', color=color)
            if show_sample_confidence:
                l_conf, r_conf = confidence_interval(its_samples.T, conf=sample_confidence, remove_nans=True)
                ax.fill_between(data.lagtimes, l_conf, r_conf, alpha=0.2, color=color)

    if show_cutoff:
        ax.plot(data.lagtimes, data.lagtimes, linewidth=2, color='black')
        ax.fill_between(data.lagtimes, np.full((data.n_lagtimes,), fill_value=ax.get_ylim()[0]), data.lagtimes,
                        alpha=0.5, color='grey')
