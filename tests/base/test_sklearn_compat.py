import unittest

import mdshare
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate

import sktime
import sktime.clustering.kmeans as kmeans
import sktime.decomposition.tica as tica
import sktime.markov.msm as msm
from sktime.markov import TransitionCountEstimator
from sktime.markov._base import TimeSeriesCVSplitter


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
            ('tica', tica.TICA(dim=1)),
            ('cluster', kmeans.KmeansClustering(n_clusters=2, max_iter=500)),
            ('counts', TransitionCountEstimator(lagtime=1, count_mode="sliding"))
        ])
        pipeline.fit(data, tica__lagtime=1)
        counts = pipeline[-1].fetch_model().submodel_largest()
        mlmsm = msm.MaximumLikelihoodMSM().fit(counts).fetch_model()
        P = mlmsm.pcca(2).coarse_grained_transition_matrix
        mindist = min(np.linalg.norm(P - transition_matrix), np.linalg.norm(P - transition_matrix.T))
        assert mindist < 0.05

    def test_cross_validation(self):
        data = sktime.data.ellipsoids().observations(1000, n_dim=50, noise=True)
        tica_model = sktime.decomposition.TICA().fit(data, lagtime=1).fetch_model()
        tica_estimator = sktime.decomposition.TICA()

        def scorer(tica_est, X, *args, **kw):
            test_model = sktime.decomposition.TICA().fit(X, lagtime=1).fetch_model()
            return tica_est.fetch_model().score(test_model=test_model)

        split = TimeSeriesCVSplitter(n_splits=10, lagtime=1, sliding=True)
        scores = cross_validate(tica_estimator, data, scoring=scorer, cv=split, fit_params=dict(lagtime=1))
        print(scores)
        print(tica_model.score())
