/* * This file is part of MSMTools.
 *
 * Copyright (c) 2015, 2014 Computational Molecular Biology Group
 *
 * MSMTools is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/* moduleauthor:: F. Paul <fabian DOT paul AT fu-berlin DOT de> */
#include <stdlib.h>
#include <math.h>
#include <string.h>
#undef NDEBUG
#include <assert.h>
#include "sigint_handler.h"
#include "_mle_trev.h"

#ifdef _MSC_VER
#undef isnan
int isnan(double var)
{
    volatile double d = var;
    return d != d;
}
#endif

static double relative_error(const int n, const double *const a, const double *const b)
{
  double sum;
  double d;
  double max = 0.0;
  int i;
  for(i=0; i<n; i++) {
    sum = 0.5*(a[i]+b[i]);
    if(sum>0) {
      d = fabs((a[i]-b[i])/sum);
      if(d>max) max=d;
    }
  }
  return max;
}

int _mle_trev_sparse(double * const T_data, const double * const CCt_data,
					const int * const i_indices, const int * const j_indices,
					const int len_CCt, const double * const sum_C,
					const int dim, const double maxerr, const int maxiter,
					double * const mu,
					double eps_mu)
{
  double rel_err;
  int i, j, t, err, iteration;
  double *sum_x, *sum_x_new, *temp;
  double CCt_ij, value;
  double x_norm;

  sigint_on();

  err = 0;

  sum_x= (double*)malloc(dim*sizeof(double));
  sum_x_new= (double*)malloc(dim*sizeof(double));
  if(!(sum_x && sum_x_new)) { err=1; goto error; }

  /* ckeck sum_C */
  for(i = 0; i<dim; i++) if(sum_C[i]==0) { err=3; goto error; }

  /* initialize sum_x_new */
  x_norm = 0;
  for(i=0; i<dim; i++) sum_x_new[i]=0;
  for(t=0; t<len_CCt; t++) {
      j = j_indices[t];
      CCt_ij = CCt_data[t];
      sum_x_new[j] += CCt_ij;
      x_norm += CCt_ij;
  }
  for(i=0; i<dim; i++) sum_x_new[i] /= x_norm;

  /* iterate */
  iteration = 0;
  do {
    /* swap buffers */
    temp = sum_x;
    sum_x = sum_x_new;
    sum_x_new = temp;

    /* update sum_x */
    for(i=0; i<dim; i++) sum_x_new[i] = 0;
    for(t=0; t<len_CCt; t++) {
      i = i_indices[t];
      j = j_indices[t];
      CCt_ij = CCt_data[t];
      value = CCt_ij / (sum_C[i]/sum_x[i] + sum_C[j]/sum_x[j]);
      sum_x_new[j] += value;
    }

    for(i = 0; i<dim; i++) if(sum_x_new[i]==0 || isnan(sum_x_new[i])) { err=2; goto error; }

    /* normalize sum_x */
    x_norm = 0;
    for(i=0; i<dim; i++) x_norm += sum_x_new[i];
    for(i=0; i<dim; i++) {
      sum_x_new[i] /= x_norm;
      if (sum_x_new[i] <= eps_mu) { err = 6; goto error; }
    }

    iteration += 1;
    rel_err = relative_error(dim, sum_x, sum_x_new);
  } while(rel_err > maxerr && iteration < maxiter && !interrupted);

  /* calculate X */
  for(i=0; i<dim; i++) sum_x[i] = 0;  // updated sum
  for(t=0; t<len_CCt; t++) {
    i = i_indices[t];
    j = j_indices[t];
    CCt_ij = CCt_data[t];
    T_data[t] = CCt_ij / (sum_C[i]/sum_x_new[i] + sum_C[j]/sum_x_new[j]);
    sum_x[i] += T_data[t];  // update sum with X_ij
  }
  /* normalize to T */
  for(t=0; t<len_CCt; t++) {
    i = i_indices[t];
    T_data[t] /= sum_x[i];
  }

  if(iteration==maxiter) { err=5; goto error; }

  memcpy(mu, sum_x_new, dim*sizeof(double));
  free(sum_x_new);
  free(sum_x);
  sigint_off();
  return 0;

error:
  memcpy(mu, sum_x_new, dim*sizeof(double));
  free(sum_x_new);
  free(sum_x);
  sigint_off();
  return -err;
}
