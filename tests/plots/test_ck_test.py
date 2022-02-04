import numpy as np

from deeptime.data import double_well_discrete
from deeptime.markov.msm import MaximumLikelihoodMSM, BayesianMSM
from deeptime.plots.chapman_kolmogorov import plot_ck_test, ChapmanKolmogorovTest, MembershipsObservable


def test_sanity():
    dtraj = double_well_discrete().dtraj_n6good
    est = MaximumLikelihoodMSM()
    est.fit(dtraj, lagtime=1)
    validator = est.chapman_kolmogorov_validator(2, mlags=10)
    cktest = validator.fit_fetch(dtraj)

    import matplotlib.pyplot as plt

    plot_ck_test(cktest)

    plt.show()

def test_sanity2():
    mlags = np.arange(2, 10)
    dtraj = double_well_discrete().dtraj_n6good
    models = []
    for lag in mlags:
        msm = MaximumLikelihoodMSM(lagtime=lag).fit_fetch(dtraj, count_mode='effective')
        bmsm = BayesianMSM().fit_fetch(msm)
        models.append(bmsm)
    memberships = models[0].prior.pcca(2).memberships
    cktest = ChapmanKolmogorovTest.from_models(models, MembershipsObservable(test_model=models[0],
                                                                             memberships=memberships))

    import matplotlib.pyplot as plt
    plot_ck_test(cktest)
    plt.show()
