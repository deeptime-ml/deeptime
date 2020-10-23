"""
Created on 28.10.2013

@author: marscher
"""
from numpy.testing import assert_allclose as assert_allclose_np

__all__ = ['assert_allclose', ]

from scipy.sparse import issparse


def assert_allclose(actual, desired, rtol=1.e-5, atol=1.e-8,
                    err_msg=''):
    r"""wrapper for numpy.testing.allclose with default tolerances of
    numpy.allclose. Needed since testing method has different values."""
    if issparse(actual):
        actual = actual.toarray()
    if issparse(desired):
        desired = desired.toarray()
    return assert_allclose_np(actual, desired, rtol=rtol, atol=atol,
                              err_msg=err_msg, verbose=True)

