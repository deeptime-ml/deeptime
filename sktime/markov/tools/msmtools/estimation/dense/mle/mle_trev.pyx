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



numpy.import_array()

import warnings

cdef extern from "_mle_trev.h":
  int _mle_trev_dense(double * const T, const double * const CCt,
                      const double * const sum_C, const int dim,
                      const double maxerr, const int maxiter,
                      double * const mu,
                      double eps_mu)


def mle_trev(C, double maxerr=1.0E-12, int maxiter=int(1.0E6),
             warn_not_converged=True, return_statdist=False,
             eps_mu=1.0E-15):
  from ....analysis import is_connected
  from ....util.exceptions import NotConvergedWarning
  assert maxerr > 0, 'maxerr must be positive'
  assert maxiter > 0, 'maxiter must be positive'
  assert C.shape[0] == C.shape[1], 'C must be a square matrix.'
  assert is_connected(C, directed=True), 'C must be strongly connected'

  cdef numpy.ndarray[double, ndim=1, mode="c"] C_sum = C.sum(axis=1).astype(numpy.float64, order='C', copy=False)
  cdef numpy.ndarray[double, ndim=2, mode="c"] CCt = (C+C.T).astype(numpy.float64, order='C', copy=False)

  cdef numpy.ndarray[double, ndim=2, mode="c"] T = numpy.zeros(C.shape, dtype=numpy.float64, order='C')
  cdef numpy.ndarray[double, ndim=1, mode="c"] mu = numpy.zeros(C.shape[0], dtype=numpy.float64, order='C')
  err = _mle_trev_dense(
        <double*> numpy.PyArray_DATA(T),
        <double*> numpy.PyArray_DATA(CCt),
        <double*> numpy.PyArray_DATA(C_sum),
        CCt.shape[0],
        maxerr,
        maxiter,
        <double*> numpy.PyArray_DATA(mu),
        eps_mu)


  if err == -1:
    raise Exception('Out of memory.')
  elif err == -2:
    raise Exception('The update of the stationary distribution produced zero or NaN.')
  elif err == -3:
    raise Exception('Some row and corresponding column of C have zero counts.')
  elif err == -5 and warn_not_converged:
    warnings.warn('Reversible transition matrix estimation didn\'t converge.',
                  NotConvergedWarning)
  elif err == -6:
    raise Exception("Stationary distribution contains entries smaller than %s during"
                    " iteration" % eps_mu)

  if return_statdist:
      return T, mu
  else:
      return T
