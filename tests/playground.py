from typing import Optional

import numpy as np

from sklearn.model_selection import GridSearchCV, BaseCrossValidator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from sktime.data import double_well_discrete, timeshifted_split
from sktime.markov import TransitionCountEstimator, TransitionCountModel
from sktime.markov.msm import MaximumLikelihoodMSM

import pandas as pd

counts = TransitionCountEstimator(lagtime=150, count_mode="sliding").fit(double_well_discrete().dtraj).fetch_model()
counts = counts.submodel_largest(directed=True)


def blocksplit_dtrajs(dtrajs, lag=1, sliding=True, shift=None):
    """ Splits the discrete trajectories into approximately uncorrelated fragments

    Will split trajectories into fragments of lengths lag or longer. These fragments
    are overlapping in order to conserve the transition counts at given lag.
    If sliding=True, the resulting trajectories will lead to exactly the same count
    matrix as when counted from dtrajs. If sliding=False (sampling at lag), the
    count matrices are only equal when also setting shift=0.

    Parameters
    ----------
    dtrajs : list of ndarray(int)
        Discrete trajectories
    lag : int
        Lag time at which counting will be done. If sh
    sliding : bool
        True for splitting trajectories for sliding count, False if lag-sampling will be applied
    shift : None or int
        Start of first full tau-window. If None, shift will be randomly generated

    """
    if not isinstance(dtrajs, (list, tuple)):
        dtrajs = [dtrajs]
    dtrajs_new = []
    for dtraj in dtrajs:
        if len(dtraj) <= lag:
            continue
        if shift is None:
            s = np.random.randint(min(lag, dtraj.size-lag))
        else:
            s = shift
        if sliding:
            if s > 0:
                dtrajs_new.append(dtraj[0:lag+s])
            for t0 in range(s, dtraj.size-lag, lag):
                dtrajs_new.append(dtraj[t0:t0+2*lag])
        else:
            for t0 in range(s, dtraj.size-lag, lag):
                dtrajs_new.append(dtraj[t0:t0+lag+1])
    return dtrajs_new

def cvsplit_dtrajs(dtrajs):
    """ Splits the trajectories into a training and test set with approximately equal number of trajectories

    Parameters
    ----------
    dtrajs : list of ndarray(int)
        Discrete trajectories

    """
    if len(dtrajs) == 1:
        raise ValueError('Only have a single trajectory. Cannot be split into train and test set')
    I0 = np.random.choice(len(dtrajs), int(len(dtrajs)/2), replace=False)
    I1 = np.array(list(set(list(np.arange(len(dtrajs)))) - set(list(I0))))
    dtrajs_train = [dtrajs[i] for i in I0]
    dtrajs_test = [dtrajs[i] for i in I1]
    return dtrajs_train, dtrajs_test

class TimeseriesBlockSplit(BaseCrossValidator):

    def __init__(self, lagtime: int, sliding:bool = True, shift: Optional[int] = None):
        if lagtime < 0:
            raise ValueError('lagtime has to be positive')
        self._lagtime = lagtime
        self._sliding = sliding
        self._shift = int(shift) if shift is not None else 0

    def split(self, X, y=None, groups=None):
        bsdt = blocksplit_dtrajs(X, lag=self._lagtime, sliding=self._sliding, shift=self._shift)
        dttrain, dttest = cvsplit_dtrajs(bsdt)
        for train, test in zip(dttrain, dttest):
            yield train, test

    def get_n_splits(self, X=None, y=None, groups=None):
        splits = [x for x in self.split(X)]
        return len(splits)



def reduce_lcc_if_fit(model: TransitionCountModel, *args, **kw):
    if model.n_states == counts.n_states_full:
        return model.submodel_largest()
    return model


estimators = [
    ('transition_counting', TransitionCountEstimator(lagtime=1, count_mode="sliding")),
    ('reduce_lcc', FunctionTransformer(reduce_lcc_if_fit)),
    ('ml_msm', MaximumLikelihoodMSM(reversible=False))
]

pipe = Pipeline(estimators)
param_grid = {
    'transition_counting__lagtime': [2, 3, 4, 5, 6, 7, 8, 9, 10]
}

scoring_fn = lambda estimator, *args, **kw: estimator['ml_msm'].fetch_model().score(double_well_discrete().dtraj, method='VAMPE')

# timeshifted_split(double_well_discrete().dtraj, lagtime=10)

gscv = GridSearchCV(pipe, param_grid, scoring=scoring_fn, cv=TimeseriesBlockSplit(lagtime=10000, shift=0))
gscv.fit(double_well_discrete().dtraj)

print(pd.DataFrame(gscv.cv_results_).to_string())

if __name__ == '__main__':
    pass
