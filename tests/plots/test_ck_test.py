from deeptime.data import double_well_discrete
from deeptime.markov.msm import MaximumLikelihoodMSM
from deeptime.plots.chapman_kolmogorov import plot_ck_test


def test_sanity():
    dtraj = double_well_discrete().dtraj_n6good
    est = MaximumLikelihoodMSM()
    est.fit(dtraj, lagtime=1)
    validator = est.chapman_kolmogorov_validator(2, mlags=10)
    cktest = validator.fit_fetch(dtraj)

    import matplotlib.pyplot as plt

    plot_ck_test(cktest)

    plt.show()

