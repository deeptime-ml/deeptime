from numpy.testing import assert_almost_equal

from deeptime.data import swissroll_model
from deeptime.markov.msm import MaximumLikelihoodMSM


def test_sanity():
    dtraj, traj = swissroll_model(100000)
    msm = MaximumLikelihoodMSM(lagtime=1).fit(dtraj).fetch_model()
    assert_almost_equal(msm.transition_matrix, swissroll_model.transition_matrix, decimal=2)
