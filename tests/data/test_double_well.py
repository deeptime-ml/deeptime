import unittest

from sktime.data import double_well_discrete
from sktime.markovprocess.msm import MarkovStateModel


class TestDoubleWell(unittest.TestCase):

    def test_cache(self):
        # load only once
        other_msm = MarkovStateModel(double_well_discrete().transition_matrix)
        assert double_well_discrete().analytic_msm is not other_msm
        assert double_well_discrete().analytic_msm is double_well_discrete().analytic_msm
