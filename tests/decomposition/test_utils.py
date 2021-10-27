import pytest
import numpy as np

from deeptime.decomposition import vamp_score_cv, VAMP


@pytest.mark.parametrize('dim', [1, 2, 3, 4, 5])
@pytest.mark.parametrize('random_state', [False, True])
@pytest.mark.parametrize('n_jobs', [1, 2])
def test_score_cv(dim, random_state, n_jobs):
    random_state = None if not random_state else np.random.RandomState(53)
    data = [np.random.uniform(size=(100, 5)) for _ in range(25)]
    estimator = VAMP(lagtime=5, dim=1)
    vamp_score_cv(estimator, data, lagtime=20, random_state=random_state, n_jobs=n_jobs)
