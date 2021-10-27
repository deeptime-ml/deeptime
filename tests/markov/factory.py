import typing

import numpy as np

import deeptime.data as datasets
from deeptime.markov import TransitionCountEstimator
from deeptime.markov.msm import BayesianMSM, MaximumLikelihoodMSM, BayesianPosterior

__all__ = ['msm_double_well', 'bmsm_double_well']


def msm_double_well(lagtime=100, reversible=True, **kwargs) -> MaximumLikelihoodMSM:
    count_model = TransitionCountEstimator(lagtime=lagtime, count_mode="sliding")\
        .fit(datasets.double_well_discrete().dtraj).fetch_model().submodel_largest()
    est = MaximumLikelihoodMSM(reversible=reversible, **kwargs)
    est.fit(count_model)
    return est


def bmsm_double_well(lagtime=100, nsamples=100, reversible=True, constrain_to_coarse_pi=False, **kwargs) -> BayesianMSM:
    """

    :param lagtime:
    :param nsamples:
    :param statdist_contraint:
    :return: tuple(Estimator, Model)
    """
    # load observations
    obs_micro = datasets.double_well_discrete().dtraj

    # stationary distribution
    pi_micro = datasets.double_well_discrete().analytic_msm.stationary_distribution
    pi_macro = np.zeros(2)
    pi_macro[0] = pi_micro[0:50].sum()
    pi_macro[1] = pi_micro[50:].sum()

    # coarse-grain microstates to two metastable states
    cg = np.zeros(100, dtype=int)
    cg[50:] = 1
    obs_macro = cg[obs_micro]

    distribution_constraint = pi_macro if constrain_to_coarse_pi else None
    counting = TransitionCountEstimator(lagtime=lagtime, count_mode="effective")\
        .fit(obs_macro).fetch_model().submodel_largest(probability_constraint=distribution_constraint)
    est = BayesianMSM(reversible=reversible, n_samples=nsamples,
                      stationary_distribution_constraint=distribution_constraint, **kwargs)
    est.fit(counting)

    return est
