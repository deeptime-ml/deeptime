import pickle
import unittest

import numpy as np

from sktime.markovprocess import MaximumLikelihoodMSM


class TestBase(unittest.TestCase):

    def test_pickle_msm(self):
        msm = MaximumLikelihoodMSM(lagtime=1, dt_traj='4 ps')
        msm.fit([np.array([0, 1, 2, 0, 1, 0, 1])],)
        msm_pickle = pickle.dumps(msm)
        assert b"version" in msm_pickle

        from numpy.testing import assert_no_warnings
        msm_restored = assert_no_warnings(pickle.loads, msm_pickle)

        # test that we can predict with the restored decision tree classifier
        model = msm.fetch_model()
        model_restored = msm_restored.fetch_model()

        np.testing.assert_equal(model_restored.transition_matrix, model.transition_matrix)
        assert model_restored.lagtime == model_restored.lagtime
        assert model.count_model.dt_traj == model_restored.count_model.dt_traj
