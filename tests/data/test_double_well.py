import numpy as np

from deeptime.data import double_well_discrete
from deeptime.markov import TransitionCountEstimator
from deeptime.markov.msm import MarkovStateModel, MaximumLikelihoodMSM


def test_cache():
    # load only once
    other_msm = MarkovStateModel(double_well_discrete().transition_matrix)
    assert double_well_discrete().analytic_msm is not other_msm
    assert double_well_discrete().analytic_msm is double_well_discrete().analytic_msm


def test_recover_timescale():
    trajs = double_well_discrete().simulate_trajectories(n_trajectories=100, n_steps=50000)
    ts = double_well_discrete().analytic_msm.timescales(1)[0]
    counts = TransitionCountEstimator(1, 'sliding').fit(trajs).fetch_model()
    msm = MaximumLikelihoodMSM().fit(counts.submodel_largest()).fetch_model()
    ts_rec = msm.timescales(1)[0]
    np.testing.assert_(np.abs(ts - ts_rec) <= 200.)
