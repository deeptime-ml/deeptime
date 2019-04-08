import unittest

import numpy as np

from sktime.markovprocess.markov_state_model import MarkovStateModel


class TestMarkovStateModel(unittest.TestCase):

    def setUp(self):
        self.msm = MarkovStateModel(np.array([[0.1, 0.9], [0.9, 0.1]]))

    def test_dt_model(self):
        self.msm.dt_model = '50 ns'
        print(self.msm.timestep_model)
