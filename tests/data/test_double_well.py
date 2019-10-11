import unittest

from sktime.datasets import double_well_discrete
from sktime.markovprocess import MarkovStateModel


class TestDoubleWell(unittest.TestCase):

    def test_cache(self):
        # load only once
        other_msm = MarkovStateModel(double_well_discrete().transition_matrix)
        assert double_well_discrete().msm is not other_msm
        assert double_well_discrete().msm is double_well_discrete().msm
