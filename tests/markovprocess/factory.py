import numpy as np
import typing

from sktime.markovprocess import BayesianMSM, MaximumLikelihoodMSM, BayesianMSMPosterior
from . import datasets

__all__ = ['data_double_well', 'msm_double_well', 'bmsm_double_well']

data_double_well = datasets.load_2well_discrete()


def bayesian_markov_model(dtrajs, lag, return_estimator=False, **kwargs) \
        -> (typing.Optional[BayesianMSM], BayesianMSMPosterior):
    est = BayesianMSM(lagtime=lag, **kwargs)
    est.fit(dtrajs)
    model = est.fetch_model()
    if return_estimator:
        return est, model
    return model


def msm_double_well(lagtime=100, reversible=True, **kwargs) -> MaximumLikelihoodMSM:
    est = MaximumLikelihoodMSM(lagtime=lagtime, reversible=reversible, **kwargs)
    est.fit(data_double_well.dtraj_T100K_dt10)
    return est


def bmsm_double_well(lagtime=100, nsamples=100, reversible=True, constrain_to_coarse_pi=False, **kwargs) -> BayesianMSM:
    """

    :param lagtime:
    :param nsamples:
    :param statdist_contraint:
    :return: tuple(Estimator, Model)
    """
    # load observations
    obs_micro = data_double_well.dtraj_T100K_dt10

    # stationary distribution
    pi_micro = data_double_well.msm.stationary_distribution
    pi_macro = np.zeros(2)
    pi_macro[0] = pi_micro[0:50].sum()
    pi_macro[1] = pi_micro[50:].sum()

    # coarse-grain microstates to two metastable states
    cg = np.zeros(100, dtype=int)
    cg[50:] = 1
    obs_macro = cg[obs_micro]

    est = BayesianMSM(lagtime=lagtime, reversible=reversible, nsamples=nsamples,
                      dt_traj='4ps',
                      statdist_constraint=pi_macro if constrain_to_coarse_pi else None,
                      **kwargs)
    est.fit(obs_macro)

    return est
