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
    its : ndarray (n_lagtimes, n_processes)
        The timescales for each process.
    its_stats : ndarray (n_lagtimes, n_processes, n_samples), optional, default=None
        Sampled timescales.

    See Also
    --------
    plot_implied_timescales
    """

    def __init__(self, lagtimes, its, its_stats=None):
        self._lagtimes = np.asarray(lagtimes, dtype=int)
        self._its = np.asarray(its)
        assert self.its.ndim == 2 and self.its.shape[0] == self.n_lagtimes, \
            "its should be of shape (lagtimes, processes)."

        if its_stats is not None:
            self._its_stats = np.asarray(its_stats).transpose(0, 2, 1)
            if not (self.its_stats.ndim == 3 and self.its_stats.shape[0] == self.n_lagtimes and
                    self.its_stats.shape[1] == self.n_processes):
                raise ValueError(f"its_stats should be of shape (lagtimes={self.n_lagtimes}, "
                                 f"processes={self.n_processes}, samples={self.n_processes}) but was "
                                 f"{self.its_stats.shape}")
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
    def its(self) -> np.ndarray:
        r""" An ndarray of shape (`n_lagtimes`, `n_processes`) containing the timescales of each process. """
        return self._its

    @property
    def its_stats(self) -> Optional[np.ndarray]:
        r""" An ndarray of shape (`n_lagtimes`, `n_processes`, `n_samples`) or representing the timescales
        of each process in each sample or `None` if no samples are available. """
        return self._its_stats

    @property
    def n_lagtimes(self) -> int:
        r""" Number of lagtimes. """
        return len(self.lagtimes)

    @property
    def n_processes(self) -> int:
        r""" Number of processes. """
        return self.its.shape[1]

    @property
    def n_samples(self) -> int:
        r""" Number of samples. """
        return 0 if self.its_stats is None else self.its_stats.shape[2]

    @staticmethod
    def from_models(models, n_its=None):
        r""" Converts a list of models to a :class:`ImpliedTimescalesData` object.

        Parameters
        ----------
        data : list of models
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

        is_bayesian = hasattr(models[0], 'prior') and hasattr(models[0], 'samples')
        lagtimes = []
        its = []
        its_stats = [] if is_bayesian else None

        for model in models:
            lagtimes.append(model.lagtime)
            if is_bayesian:
                result = model.timescales(k=n_its)
                its.append(result[0])
                its_stats.append(result[1])
            else:
                its.append(model.timescales(k=n_its))
        return ImpliedTimescalesData(lagtimes, its, its_stats)


@plotting_function
def plot_implied_timescales(ax, data, n_its: Optional[int] = None, process: Optional[int] = None,
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
    if process is not None and process >= data.n_processes:
        raise ValueError(f"Requested process {process} when only {data.n_processes} are available.")

    if process is None and n_its is None:
        n_its = data.n_processes
    it_indices = [process] if process is not None else np.arange(n_its)
    if colors is None:
        from matplotlib import rcParams
        colors = rcParams['axes.prop_cycle'].by_key()['color']
    for it_index in it_indices:
        color = colors[it_index % len(colors)]
        if show_mle:
            ax.plot(data.lagtimes, data.its[:, it_index], color=color, **kwargs)
        if data.n_samples > 0 and show_samples:
            if show_sample_mean:
                sample_mean = np.mean(data.its_stats[:, it_index], axis=1)
                ax.plot(data.lagtimes, sample_mean, marker='o', linestyle='dashed', color=color)
            if show_sample_confidence:
                l_conf, r_conf = confidence_interval(data.its_stats[:, it_index].T, conf=sample_confidence)
                ax.fill_between(data.lagtimes, l_conf, r_conf, alpha=0.2, color=color)

    if show_cutoff:
        ax.plot(data.lagtimes, data.lagtimes, linewidth=2, color='black')
        ax.fill_between(data.lagtimes, np.full((data.n_lagtimes,), fill_value=ax.get_ylim()[0]), data.lagtimes,
                        alpha=0.5, color='grey')
