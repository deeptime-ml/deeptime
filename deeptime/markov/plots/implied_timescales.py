from typing import Optional

import numpy as np

from deeptime.markov import BayesianPosterior
from deeptime.markov.hmm import HiddenMarkovModel, BayesianHMMPosterior
from deeptime.markov.msm import MarkovStateModel
from deeptime.util import confidence_interval
from deeptime.util.decorators import plotting_function


class ImpliedTimescalesData:

    def __init__(self, lagtimes, its, its_stats=None):
        self._lagtimes = np.asarray(lagtimes, dtype=int)
        self._its = np.asarray(its)
        assert self.its.ndim == 2 and self.its.shape[0] == self.n_lagtimes, \
            "its should be of shape (lagtimes, timescales)."

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
        return self._lagtimes

    @property
    def its(self) -> np.ndarray:
        return self._its

    @property
    def its_stats(self) -> Optional[np.ndarray]:
        return self._its_stats

    @property
    def n_lagtimes(self) -> int:
        return len(self.lagtimes)

    @property
    def n_processes(self) -> int:
        return self.its.shape[1]

    @property
    def n_samples(self) -> int:
        return 0 if self.its_stats is None else self.its_stats.shape[2]


allowed_types = [MarkovStateModel, BayesianPosterior,
                 HiddenMarkovModel, BayesianHMMPosterior]


def to_its_data(data, n_its=None) -> ImpliedTimescalesData:
    if isinstance(data, ImpliedTimescalesData):
        return data
    elif isinstance(data, (list, tuple)):
        if len(data) == 0:
            raise ValueError("Data cannot be empty.")
        ix = -1
        for i, allowed_type in enumerate(allowed_types):
            if isinstance(data[0], allowed_type):
                ix = i
                break
        if ix == -1:
            raise ValueError(f"If provided as a list of models, the contained elements must all "
                             f"be of type {[x.__name__ for x in allowed_types]}.")
        selected_type = allowed_types[ix]
        if not all(isinstance(x, selected_type) for x in data):
            raise ValueError(f"If provided as a list of models, the contained elements must all be of the same type. "
                             f"The first element was a {selected_type.__name__}, which does not agree with the rest.")
        # now we have made sure, that all models are of the same type...
        is_bayesian = isinstance(data[0], BayesianPosterior)
        lagtimes = []
        its = []
        its_stats = [] if is_bayesian else None

        for model in data:
            lagtimes.append(model.lagtime)
            if is_bayesian:
                result = model.timescales(k=n_its)
                its.append(result[0])
                its_stats.append(result[1])
            else:
                its.append(model.timescales(k=n_its))
        return ImpliedTimescalesData(lagtimes, its, its_stats)
    else:
        raise ValueError(f"Unknown type of data. ImpliedTimescalesData or list/tuple of MSM objects is allowed, "
                         f"got {data} instead.")


@plotting_function
def plot_implied_timescales(ax, data, n_its: Optional[int] = None, process: Optional[int] = None,
                            show_mle: bool = True, show_samples: bool = True, show_sample_mean: bool = True,
                            show_sample_confidence: bool = True, show_cutoff: bool = True,
                            sample_confidence: float = .95,
                            colors=None, **kwargs):
    if n_its is not None and process is not None:
        raise ValueError("n_its and process are mutually exclusive.")
    data = to_its_data(data, n_its)
    if process is not None and process >= data.n_processes:
        raise ValueError(f"Requested process {process} when only {data.n_processes} are available.")

    if process is None and n_its is None:
        n_its = data.n_processes
    it_indices = [process] if process is not None else np.arange(n_its)
    if colors is None:
        colors = [f"C{i}" for i in range(len(it_indices))]
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
        ax.fill_between(data.lagtimes, ax.get_ylim()[0]*np.ones(data.n_lagtimes), data.lagtimes,
                        alpha=0.5, color='grey')

