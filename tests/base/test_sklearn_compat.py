import unittest
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import mdshare
import numpy as np
from sklearn.pipeline import Pipeline

import deeptime.clustering.kmeans as kmeans
import deeptime.decomposition.tica as tica
import deeptime.markov.msm as msm
from deeptime.markov import TransitionCountEstimator


class TestSkLearnCompat(unittest.TestCase):
    """
    pipelining
    cross validation
    """

    def test_mlmsm_pipeline(self):
        file = mdshare.fetch('hmm-doublewell-2d-100k.npz', working_directory='data')

        with np.load(file) as fh:
            data = fh['trajectory']
            transition_matrix = fh['transition_matrix']

        pipeline = Pipeline(steps=[
            ('tica', tica.TICA(dim=1, lagtime=1)),
            ('cluster', kmeans.KmeansClustering(n_clusters=2, max_iter=500)),
            ('counts', TransitionCountEstimator(lagtime=1, count_mode="sliding"))
        ])
        pipeline.fit(data)
        counts = pipeline[-1].fetch_model().submodel_largest()
        mlmsm = msm.MaximumLikelihoodMSM().fit(counts).fetch_model()
        P = mlmsm.pcca(2).coarse_grained_transition_matrix
        mindist = min(np.linalg.norm(P - transition_matrix), np.linalg.norm(P - transition_matrix.T))
        assert mindist < 0.05
