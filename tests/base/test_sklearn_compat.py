import unittest

import mdshare
import numpy as np
from sklearn.pipeline import Pipeline

import sktime.clustering.kmeans as kmeans
import sktime.decomposition.tica as tica
import sktime.markovprocess.maximum_likelihood_msm as msm
from sktime.data.double_well import DoubleWellDiscrete
from sktime.markovprocess import TransitionCountEstimator
from sktime.markovprocess.maximum_likelihood_hmsm import MaximumLikelihoodHMSM


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
            ('counts', TransitionCountEstimator(lagtime=1, count_mode="sliding"))
        ])
        pipeline.fit(data)
        counts = pipeline[-1].fetch_model().submodel_largest()
        mlmsm = msm.MaximumLikelihoodMSM().fit(counts).fetch_model()
        P = mlmsm.pcca(2).coarse_grained_transition_matrix
        mindist = min(np.linalg.norm(P - transition_matrix), np.linalg.norm(P - transition_matrix.T))
        assert mindist < 0.05

    def test_hmm_stuff(self):
        obs = DoubleWellDiscrete().dtraj.copy()
        obs -= np.min(obs)  # remove empty states
        hmsm = MaximumLikelihoodHMSM(n_states=2, lagtime=1)
        model = hmsm.fit([obs, obs]).fetch_model()
        print(model)
