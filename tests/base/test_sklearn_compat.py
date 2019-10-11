import unittest

import numpy as np

import mdshare
from sklearn.pipeline import Pipeline

import sktime.decomposition.tica as tica
import sktime.clustering.kmeans as kmeans
import sktime.markovprocess.maximum_likelihood_msm as msm
import sktime.markovprocess.pcca as pcca


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
            ('tica', tica.TICA(lagtime=1, dim=1)),
            ('cluster', kmeans.KmeansClustering(n_clusters=2, max_iter=500)),
            ('msm', msm.MaximumLikelihoodMSM())
        ])
        pipeline.fit(data)
        mlmsm = pipeline[-1].fetch_model()
        P = pcca.PCCAEstimator(n_metastable=2).fit(mlmsm)\
            .fetch_model().coarse_grained_transition_matrix
        mindist = min(np.linalg.norm(P - transition_matrix), np.linalg.norm(P - transition_matrix.T))
        assert mindist < 0.05
