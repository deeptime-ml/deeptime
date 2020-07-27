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
    d += (a[i]-b[i])*(a[i]-b[i]);
  }
  return d;
}

#define C(i,j) (C [(i)*n+(j)])
#define T(i,j) (T[(i)*n+(j)])

int _mle_trev_given_pi_dense(double * const T, const double * const C, const double * const mu, const int n, const double maxerr, const int maxiter)
{
  double d_sq, norm, C_ij;
  int i, j, err, iteration;
  double *lam, *lam_new, *temp;

  sigint_on();

  lam= (double*)malloc(n*sizeof(double));
  lam_new= (double*)malloc(n*sizeof(double));
  if(!(lam && lam_new)) { err=1; goto error; }

  /* check mu */
  for(i=0; i<n; i++) {
    if(mu[i]==0) { err=4; goto error; }
  }

  /* initialise lambdas */
  for(i=0; i<n; i++) {
    lam_new[i] = 0.0;
    for(j=0; j<n; j++) {
      lam_new[i] += 0.5*(C(i,j)+C(j,i));
    }
    if(lam_new[i]==0) { err=3; goto error; }
  }

  /* iterate lambdas */
  iteration = 0;
  do {
    /* swap buffers */
    temp = lam;
    lam = lam_new;
    lam_new = temp;

    err = 0;

#pragma omp parallel for private(i,C_ij)
    for(j=0; j<n; j++) {
      lam_new[j] = 0.0;
      for(i=0; i<n; i++) {
        C_ij = C(i,j)+C(j,i);
        if(C_ij==0) continue;
        lam_new[j] += C_ij / ((mu[j]*lam[i])/(mu[i]*lam[j])+1);
      }
      if(isnan(lam_new[j]) && err==0) err=2;
    }

    if(err!=0) goto error;
    iteration += 1;
    d_sq = distsq(n,lam,lam_new);
  } while(d_sq > maxerr*maxerr && iteration < maxiter && !interrupted);

  /* calculate T */
  for(i=0; i<n; i++) {
    norm = 0;
    // printf("%e\n", lam_new[i]);
    for(j=0; j<n; j++) {
      C_ij = C(i,j)+C(j,i);
      if(i!=j) {
	if (C_ij>0.0){
	  T(i,j) = C_ij / (lam_new[i] + lam_new[j]*mu[i]/mu[j]);
	// printf("%f ", T(i, j));
	  norm += T(i,j);
	}
	else{
	  T(i,j) = 0.0;
	}
      }
    }
    // printf("\n");
    if(norm>1.0) T(i,i) = 0.0; else T(i,i) = 1.0-norm;
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
