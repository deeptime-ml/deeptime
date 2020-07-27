r"""Cython implementation of iterative likelihood maximization.

.. moduleauthor:: F. Paul <fabian DOT paul AT fu-berlin DOT de>

"""

import numpy
import scipy
import scipy.sparse
cimport numpy
numpy.import_array()

import msmtools.estimation
import warnings
import msmtools.util.exceptions

cdef extern from "_mle_trev.h":
  int _mle_trev_sparse(double * const T_data, const double * const CCt_data, 
                       const int * const i_indices, const int * const j_indices,
                       const int len_CCt, const double * const sum_C, const int dim,
                       const double maxerr, const int maxiter,
                       double * const mu,
                       double eps_mu)


def mle_trev(C, double maxerr=1.0E-12, int maxiter=int(1.0E6),
             warn_not_converged=True, return_statdist=False,
             eps_mu=1.0E-15):

  assert maxerr > 0, 'maxerr must be positive'
  assert maxiter > 0, 'maxiter must be positive'
  assert C.shape[0] == C.shape[1], 'C must be a square matrix.'
  assert msmtools.estimation.is_connected(C, directed=True), 'C must be strongly connected'

  C_sum_py = C.sum(axis=1).A1
  cdef numpy.ndarray[double, ndim=1, mode="c"] C_sum = C_sum_py.astype(numpy.float64, order='C', copy=False)

  CCt = C+C.T
  # convert CCt to coo format 
  CCt_coo = CCt.tocoo()
  n_data = CCt_coo.nnz
  cdef numpy.ndarray[double, ndim=1, mode="c"] CCt_data =  CCt_coo.data.astype(numpy.float64, order='C', copy=False)
  cdef numpy.ndarray[int, ndim=1, mode="c"] i_indices = CCt_coo.row.astype(numpy.intc, order='C', copy=True)
  cdef numpy.ndarray[int, ndim=1, mode="c"] j_indices = CCt_coo.col.astype(numpy.intc, order='C', copy=True)

  # prepare data array of T in coo format
  cdef numpy.ndarray[double, ndim=1, mode="c"] T_data = numpy.zeros(n_data, dtype=numpy.float64, order='C')
  cdef numpy.ndarray[double, ndim=1, mode="c"] mu = numpy.zeros(C.shape[0], dtype=numpy.float64, order='C')
  err = _mle_trev_sparse(
        <double*> numpy.PyArray_DATA(T_data),
        <double*> numpy.PyArray_DATA(CCt_data),
        <int*> numpy.PyArray_DATA(i_indices),
        <int*> numpy.PyArray_DATA(j_indices),
        n_data,
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
                  msmtools.util.exceptions.NotConvergedWarning)
  elif err == -6:
    raise Exception("Stationary distribution contains entries smaller than %s during"
                    " iteration" % eps_mu)
  # T matrix has the same shape and positions of nonzero elements as CCt
  T = scipy.sparse.csr_matrix((T_data, (i_indices, j_indices)), shape=CCt.shape)
  T = msmtools.estimation.sparse.transition_matrix.correct_transition_matrix(T)
  if return_statdist:
      return T, mu
  else:
      return T
