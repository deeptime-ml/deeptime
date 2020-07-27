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
#ifdef _MSC_VER
#undef isnan
static int isnan(double var)
{
    volatile double d = var;
    return d != d;
}
#endif

#undef NDEBUG
#include <assert.h>
#include "sigint_handler.h"
#include "_mle_trev.h"

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

#define CCt(i,j) (CCt[(i)*dim+(j)])
#define T(i,j) (T[(i)*dim+(j)])

int _mle_trev_dense(double * const T, const double * const CCt,
                    const double * const sum_C, const int dim,
                    const double maxerr, const int maxiter,
                    double * const mu,
                    double eps_mu)
{
  double rel_err, x_norm;
  int i, j, err, iteration;
  double *sum_x, *sum_x_new, *temp;

  sigint_on();

  sum_x= (double*)malloc(dim*sizeof(double));
  sum_x_new= (double*)malloc(dim*sizeof(double));
  if(!(sum_x && sum_x_new)) { err=1; goto error; }

  /* ckeck sum_C */
  for(i = 0; i<dim; i++) if(sum_C[i]==0) { err=3; goto error; }

  /* initialize sum_x_new */
  x_norm = 0;
  for(i=0; i<dim; i++) {
    sum_x_new[i]=0;
	for(j=0; j<dim; j++) {
	   sum_x_new[i] += CCt(i,j);
	}
	x_norm += sum_x_new[i];
  }
  for(i=0; i<dim; i++) sum_x_new[i] /= x_norm;

  /* iterate */
  iteration = 0;
  do {
    /* swap buffers */
    temp = sum_x;
    sum_x = sum_x_new;
    sum_x_new = temp;

	x_norm = 0;
    for(i=0; i<dim; i++) {
      sum_x_new[i] = 0;
	  for(j=0; j<dim; j++) {
         sum_x_new[i] += CCt(i,j) / (sum_C[i]/sum_x[i] + sum_C[j]/sum_x[j]);
	  }
	  if(sum_x_new[i]==0 || isnan(sum_x_new[i])) { err=2; goto error; }
	  x_norm += sum_x_new[i];
    }

    /* normalize sum_x */
    for(i=0; i<dim; i++) {
      sum_x_new[i] /= x_norm;
      if (sum_x_new[i] <= eps_mu) { err = 6; goto error; }
    }

    iteration += 1;
    rel_err = relative_error(dim, sum_x, sum_x_new);
  } while(rel_err > maxerr && iteration < maxiter && !interrupted);

  /* calculate T*/
  for(i=0; i<dim; i++) {
    sum_x[i] = 0;  // updated sum
    for(j=0; j<dim; j++) {
      T(i,j) = CCt(i,j) / (sum_C[i]/sum_x_new[i] + sum_C[j]/sum_x_new[j]);  // X_ij
      sum_x[i] += T(i,j);  // update sum with X_ij
	}
	/* normalize X to T*/
    for(j=0; j<dim; j++) {
      T(i,j) /= sum_x[i];
    }
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

#undef T
#undef CCt
