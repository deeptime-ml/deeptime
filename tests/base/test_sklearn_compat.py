import unittest

import numpy as np
from sklearn.pipeline import Pipeline

from deeptime.clustering import KMeans
from deeptime.decomposition import TICA
from deeptime.markov import TransitionCountEstimator
from deeptime.markov.hmm import HiddenMarkovModel, GaussianOutputModel
from deeptime.markov.msm import MaximumLikelihoodMSM, MarkovStateModel


class TestSkLearnCompat(unittest.TestCase):
    """
    pipelining
    cross validation
    """

    def test_mlmsm_pipeline(self):
        hmm = HiddenMarkovModel(transition_model=MarkovStateModel([[.8, .2], [.1, .9]]),
                                output_model=GaussianOutputModel(n_states=2, means=[-10, 10], sigmas=[.1, .1]))
        htraj, traj = hmm.simulate(10000)
        transition_matrix = hmm.transition_model.transition_matrix
        pipeline = Pipeline(steps=[
            ('tica', TICA(dim=1, lagtime=1)),
            ('cluster', KMeans(n_clusters=2, max_iter=500)),
            ('counts', TransitionCountEstimator(lagtime=1, count_mode="sliding"))
        ])
        pipeline.fit(traj[..., None])
        counts = pipeline[-1].fetch_model().submodel_largest()
        mlmsm = MaximumLikelihoodMSM().fit(counts).fetch_model()
        P = mlmsm.pcca(2).coarse_grained_transition_matrix
        mindist = min(np.linalg.norm(P - transition_matrix), np.linalg.norm(P - transition_matrix.T))
        assert mindist < 0.05
