from numpy.testing import assert_almost_equal, assert_

import deeptime as dt
import numpy as np


def test_recover_model(fixed_seed):
    dtraj, traj = dt.data.sqrt_model(500000, seed=42)
    msm_rec = dt.markov.msm.MaximumLikelihoodMSM().fit(dtraj, lagtime=1).fetch_model()
    assert_almost_equal(msm_rec.transition_matrix, dt.data.sqrt_model.transition_matrix, decimal=3)
    traj[:, 1] = traj[:, 1] - np.sqrt(np.abs(traj[:, 0]))  # inverse transformation
    assert_(traj[np.argwhere(dtraj == 1)[0], 1] <= 0)
    assert_(traj[np.argwhere(dtraj == 0)[0], 1] >= 0)
