
# This file is part of MSMTools.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# MSMTools is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

r"""Cython implementation of iterative likelihood maximization.

.. moduleauthor:: F. Paul <fabian DOT paul AT fu-berlin DOT de>

"""

import numpy
cimport numpy
import warnings

from ....util.exceptions import NotConvergedWarning
from ....analysis import is_connected


cdef extern from "_mle_trev_given_pi.h":
  int _mle_trev_given_pi_dense(double * const T, const double * const C, const double * const mu, const int n, const double maxerr, const int maxiter)

def mle_trev_given_pi(
  C,
  mu,
  double maxerr = 1.0E-12,
  int maxiter = 1000000,
  double eps = 0.0
  ):

  assert maxerr > 0, 'maxerr must be positive'
  assert maxiter > 0, 'maxiter must be positive'
  assert eps >= 0, 'eps must be non-negative'
  if eps>0:
     warnings.warn('A regularization parameter value eps!=0 is not necessary for convergence. The parameter will be removed in future versions.', DeprecationWarning)
  assert is_connected(C, directed=False), 'C must be (weakly) connected'

  cdef numpy.ndarray[double, ndim=2, mode="c"] c_C = C.astype(numpy.float64, order='C', copy=False)
  cdef numpy.ndarray[double, ndim=1, mode="c"] c_mu = mu.astype(numpy.float64, order='C', copy=False)

  assert c_C.shape[0]==c_C.shape[1]==c_mu.shape[0], 'Dimensions of C and mu don\'t agree.'

  cdef numpy.ndarray[double, ndim=2, mode="c"] T = numpy.zeros_like(c_C, dtype=numpy.float64, order='C')

  err = _mle_trev_given_pi_dense(
        <double*> numpy.PyArray_DATA(T),
        <double*> numpy.PyArray_DATA(c_C),
        <double*> numpy.PyArray_DATA(c_mu),
        c_C.shape[0],
        maxerr,
        maxiter)

  if err == -1:
    raise Exception('Out of memory.')
  elif err == -2:
    raise Exception('The update of the Lagrange multipliers produced NaN.')
  elif err == -3:
    raise Exception('Some row and corresponding column of C have zero counts.')
  elif err == -4:
    raise Exception('Some element of pi is zero.')
  elif err == -5:
    warnings.warn('Reversible transition matrix estimation with fixed stationary distribution didn\'t converge.', NotConvergedWarning)

  return T

