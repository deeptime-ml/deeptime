from typing import Optional

import numpy as np

from deeptime.plots.util import default_colors
from deeptime.util import confidence_interval
from deeptime.util.decorators import plotting_function
from deeptime.util.validation import ImpliedTimescales


@plotting_function()
def plot_implied_timescales(data: ImpliedTimescales, n_its: Optional[int] = None, process: Optional[int] = None,
                            show_mle: bool = True, show_samples: bool = True, show_sample_mean: bool = True,
                            show_sample_confidence: bool = True, show_cutoff: bool = True,
                            sample_confidence: float = .95,
                            colors=None, ax=None, **kwargs):
    r"""Creates an implied timescales plot inside exising matplotlib axes.

    .. plot:: examples/plot_implied_timescales.py

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The matplotlib axes to use for plotting.
    data : ImpliedTimescales
        A timescales data container object, can be obtained, e.g., via :meth:`ImpliedTimescales.from_models`.
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

    Returns
    -------
    ax : matplotlib.axes.Axes
        The matplotlib axes that were used to plot the timescales.

    See Also
    --------
    ImpliedTimescales
    """
    if ax is None:
        import matplotlib.pyplot as plt
        ax = plt.gca()
    if n_its is not None and process is not None:
        raise ValueError("n_its and process are mutually exclusive.")
    if process is not None and process >= data.max_n_processes:
        raise ValueError(f"Requested process {process} when only {data.max_n_processes} are available.")

    if process is None and n_its is None:
        n_its = data.max_n_processes
    it_indices = [process] if process is not None else np.arange(n_its)
    if colors is None:
        colors = default_colors()
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
    return ax
