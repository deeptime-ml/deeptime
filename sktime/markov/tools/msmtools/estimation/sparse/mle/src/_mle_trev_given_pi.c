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

#ifdef _MSC_VER
#undef isnan
int isnan(double var)
{
    volatile double d = var;
    return d != d;
}
#endif

#undef NDEBUG
#include <assert.h>
#include "sigint_handler.h"
#include "_mle_trev_given_pi.h"

static double distsq(const int n, const double *const a, const double *const b)
{
  double d = 0.0;
  int i;
#pragma omp parallel for reduction(+:d)
  for(i=0; i<n; i++) {
    d = d + (a[i]-b[i])*(a[i]-b[i]);
  }
  return d;
}

int _mle_trev_given_pi_sparse(
		double * const T_unnormalized_data,
		const double * const CCt_data,
		const int * const i_indices,
		const int * const j_indices,
		const int len_CCt,
		const double * const mu,
		const int len_mu,
		const double maxerr,
		const int maxiter)
{
  double d_sq;
  int i, j, t, err, iteration;
  double *lam, *lam_new, *temp;
  double CCt_ij;

  sigint_on();

  err = 0;

  lam= (double*)malloc(len_mu*sizeof(double));
  lam_new= (double*)malloc(len_mu*sizeof(double));
  if(!(lam && lam_new)) { err=1; goto error; }

  /* check mu */
  for(i=0; i<len_mu; i++) {
    if(mu[i]==0) { err=4; goto error; }
  }

  /* initialise lambdas */
  for(i=0; i<len_mu; i++) lam_new[i] = 0.0;
  for(t=0; t<len_CCt; t++) {
    i = i_indices[t];
    j = j_indices[t];
    if(i<j) continue;
    lam_new[i] += 0.5*CCt_data[t];
    if(i!=j)
      lam_new[j] += 0.5*CCt_data[t];
  }
  for(i=0; i<len_mu; i++) if(lam_new[i]==0) { err=3; goto error; }

  /* iterate lambdas */
  iteration = 0;
  do {
    /* swap buffers */
    temp = lam;
    lam = lam_new;
    lam_new = temp;

    for(i=0; i<len_mu; i++) {
       lam_new[i] = 0.0;
    }
    for(t=0; t<len_CCt; t++) {
      i = i_indices[t];
      j = j_indices[t];
      if(i<j) continue;
      CCt_ij = CCt_data[t];
      assert(CCt_ij!=0); /* should never fail */
      lam_new[i] += CCt_ij / ((mu[i]*lam[j])/(mu[j]*lam[i])+1.0);
      if(i!=j)
        lam_new[j] += CCt_ij / ((mu[j]*lam[i])/(mu[i]*lam[j])+1.0);
    }
    for(i=0; i<len_mu; i++) {
       if(isnan(lam_new[i])) { err=2; goto error; }
    }

    iteration += 1;
    d_sq = distsq(len_mu,lam,lam_new);
  } while(d_sq > maxerr*maxerr && iteration < maxiter && !interrupted);

  /* calculate T */
  for(t=0; t<len_CCt; t++) {
      i = i_indices[t];
      j = j_indices[t];
      if(i==j) T_unnormalized_data[t] = 0; /* handle normalization later */
      else {
        CCt_ij = CCt_data[t];
        T_unnormalized_data[t] = CCt_ij / (lam_new[i] + lam_new[j]*mu[i]/mu[j]);
      }
  }

  if(iteration==maxiter) { err=5; goto error; }

  free(lam);
  free(lam_new);
  sigint_off();
  return 0;

error:
  free(lam);
  free(lam_new);
  sigint_off();
  return -err;
}
