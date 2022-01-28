from typing import Optional

import numpy as np

from deeptime.markov import BayesianPosterior
from deeptime.markov.hmm import HiddenMarkovModel, BayesianHMMPosterior
from deeptime.markov.msm import MarkovStateModel
from deeptime.util.decorators import plotting_function


class ImpliedTimescalesData:

    def __init__(self, lagtimes, its, its_stats=None):
        self._lagtimes = np.asarray(lagtimes, dtype=int)
        self._its = np.asarray(its)
        assert self.its.ndim == 2 and self.its.shape[0] == self.n_lagtimes, \
            "its should be of shape (lagtimes, timescales)."

        if its_stats is not None:
            self._its_stats = np.asarray(its_stats)
            assert self.its_stats.ndim == 3 and \
                   self.its_stats.shape[0] == self.n_lagtimes and \
                   self.its_stats.shape[1] == self.n_processes, "its_stats should be of " \
                                                                "shape (lagtimes, timescales, samples)"
        else:
            self._its_stats = None

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
def implied_timescales(ax, data, n_its: Optional[int] = None):
    data = to_its_data(data, n_its)

