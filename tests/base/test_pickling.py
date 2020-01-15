import pickle
import unittest
import unittest.mock as mock

import numpy as np

from tests.markovprocess import factory


class TestPickling(unittest.TestCase):

    def test_pickle_msm(self):
        msm = factory.msm_double_well()
        msm_pickle = pickle.dumps(msm)
        assert b"version" in msm_pickle

        from numpy.testing import assert_no_warnings
        msm_restored = assert_no_warnings(pickle.loads, msm_pickle)

        # test that we can predict with the restored decision tree classifier
        model = msm.fetch_model()
        model_restored = msm_restored.fetch_model()

        np.testing.assert_equal(model_restored.transition_matrix, model.transition_matrix)
        assert model_restored.lagtime == model_restored.lagtime
        assert model.count_model.physical_time == model_restored.count_model.physical_time

    def test_pickle_bmsm(self):
        msm = factory.bmsm_double_well(nsamples=10)
        msm_pickle = pickle.dumps(msm)
        assert b"version" in msm_pickle

        from numpy.testing import assert_no_warnings
        msm_restored = assert_no_warnings(pickle.loads, msm_pickle)

        # test that we can predict with the restored decision tree classifier
        model = msm.fetch_model()
        model_restored = msm_restored.fetch_model()

        np.testing.assert_equal(model_restored.prior.transition_matrix, model.prior.transition_matrix)
        assert model_restored.prior.lagtime == model_restored.prior.lagtime
        assert model.prior.count_model.physical_time == model_restored.prior.count_model.physical_time

    def test_old_version_raise_warning(self):
        """ ensures that a user warning is displayed, when restoring an object stored with an old version.

        """
        msm = factory.msm_double_well()
        pickled = pickle.dumps(msm)
        # now simulate a newer version
        with mock.patch('sktime.__version__', '99+brand-new'), np.testing.assert_warns(UserWarning, ):
            pickle.loads(pickled)
