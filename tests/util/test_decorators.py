import pytest
import numpy as np
from numpy.testing import assert_raises, assert_warns

from deeptime.decomposition import VAMP
from deeptime.markov.hmm import MaximumLikelihoodHMM, BayesianHMM, HiddenMarkovModel
from deeptime.markov.msm import MaximumLikelihoodMSM, BayesianMSM
from deeptime.util.decorators import deprecated_argument


@deprecated_argument("arg3", "arg5", "arg3 is deprecated and replaced by arg5")
def function_with_deprecated_args(arg1=None, arg2=None, arg3=None, arg4=None, arg5=None):
    pass


def test_deprecation_warning():
    function_with_deprecated_args()

    with assert_raises(ValueError):
        function_with_deprecated_args(arg3=5, arg5=7)
    with assert_warns(DeprecationWarning):
        function_with_deprecated_args(arg3=5)
    function_with_deprecated_args(arg5=3)
