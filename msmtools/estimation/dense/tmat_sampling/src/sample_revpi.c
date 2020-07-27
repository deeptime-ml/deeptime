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

/* * moduleauthor:: B. Trendelkamp-Schroer 
 * <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>  
 */
#include <stdlib.h>
#include <stddef.h>
#include <math.h>
#include <float.h>

#include "rnglib/ranlib.h"
#include "sample_revpi.h"
#include "util.h"


double
f(double v, double s, double a1, double a2, double a3)
{
  double r;
  r = s/(s-1.0);
  return (a1+1.0)*log(v) + a3*log(r + v) - (a1+a2+a3+2.0)*log(1.0+v);
  
}

double 
F(double v, double s, double a1, double a2, double a3)
{
  double r;
  r = s/(s-1.0);
  return (a1+1.0)/v + a3/(r+v) - (a1+a2+a3+2)/(1.0+v);
}

double
DF(double v, double s, double a1, double a2, double a3)
{
  double r;
  r = s/(s-1.0);
  return -(a1+1.0)/(v*v) - a3/((r+v)*(r+v)) + (a1+a2+a3+2)/((1.0+v)*(1.0+v));
}

double 
maximum_point(double s, double a1, double a2, double a3)
{
  double a, b, c, vbar;
  a = a2 + 1.0;
  b = a2 - a1 + (a2+a3+1.0)/(s-1.0);
  c = (a1+1.0) * s/(1.0-s);
  vbar = (-b + sqrt(b*b - 4.0*a*c))/(2.0*a);
  return vbar;  
}

double
qacc(double w, double v, double s,
     double a1, double a2, double a3,
     double alpha, double beta)
{
  double r;
  r = s/(s-1.0);
  return beta * (w-v) + (a1 + 1.0 - alpha) * log(w/v) + a3 * log((r+w)/(r+v)) -
    (a1 + a2 + a3 + 2.0) * log((1.0+w)/(1.0+v));  
}

double
qacc_rw(double w, double v, double s, double a1, double a2, double a3)
{
  double r;
  r = s/(s-1.0);
  return (a1 + 1.0) * log(w/v) + a3 * log((r+w)/(r+v))
    -(a1 + a2 + a3 + 2.0) * log((1.0+w)/(1.0+v));
}

double 
sample_quad(double xkl, double xkk, double xll,
	    double ckl, double clk, double ckk, double cll,
	    double bk, double bl)
{
  double xlk, skl, slk, s2, s3, s, a1, a2, a3;
  double vbar, alpha, beta, v, w;
  double q, U;

  xlk = xkl;

  skl = xkk + xkl;
  slk = xll + xlk;
  
  if (skl <= slk)
    {
      s2 = skl;
      s3 = slk;
      s = s3/s2;
      a1 = ckl + clk - 1.0;
      a2 = ckk + bk - 1.0;
      a3 = cll + bl -1.0;	
    }
  else
    {
      s2 = slk;
      s3 = skl;
      s = s3/s2;
      a1 = ckl + clk - 1.0;
      a2 = cll + bl - 1.0;
      a3 = ckk + bk - 1.0;
    }

  //Check if s-1>0
  if (is_positive(s-1.0))
    {
      vbar = maximum_point(s, a1, a2, a3);
      beta = -1.0 * DF(vbar, s, a1, a2, a3) * vbar;
      alpha = beta * vbar;
      
      //Check if s2-xkl > 0
      if (is_positive(s2-xkl))
	{
	  //Old sample
	  v = xkl/(s2-xkl);	  
	  
	  //Check if alpha > 0 and 1/beta > 0
	  if(is_positive(alpha) && is_positive(1.0/beta))
	    {
	      //Proposal
	      w = 1.0/beta*sgamma(alpha);

	      //If w=0 -> reject
	      if(is_positive(w))
		{
		  // If v=0 accept
		  if(!is_positive(v))
		    {
		      return s2*w/(1.0+w);	
		    }
		  else
		    {
		      // Log acceptance probability
		      q = qacc(w, v, s, a1, a2, a3, alpha, beta);
		      
		      // Metropolis step
		      U = genunf(0.0, 1.0);
		      if(log(U) < my_fmin(0.0, q))
			{
			  return s2*w/(1.0+w);
			}
		    }
		}
	    }
	}
    }
  return xkl;  
}

double
sample_quad_rw(double xkl, double xkk, double xll,
	       double ckl, double clk, double ckk, double cll,
	       double bk, double bl)
{
  double xlk, skl, slk, s2, s3, s, a1, a2, a3;
  double v, w;
  double q, U;
  
  xlk = xkl;

  skl = xkk + xkl;
  slk = xll + xlk;
  
  if (skl <= slk)
    {
      s2 = skl;
      s3 = slk;
      s = s3/s2;
      a1 = ckl + clk - 1.0;
      a2 = ckk + bk - 1.0;
      a3 = cll + bl -1.0;	
    }
  else
    {
      s2 = slk;
      s3 = skl;
      s = s3/s2;
      a1 = ckl + clk - 1.0;
      a2 = cll + bl - 1.0;
      a3 = ckk + bk - 1.0;
    }
  //Check if s2-xkl > 0
  if(is_positive(s2-xkl))
    {
      //Old sample
      v = xkl/(s2-xkl);	  
      //Proposal
      w = v * exp(snorm());
      //If w=0 -> reject
      if(is_positive(w))
	{
	  //If v=0 accept
	  if(!is_positive(v))
	    {
	      return s2*w/(1.0+w);
	    }
	  else
	    {
	      q = qacc_rw(w, v, s, a1, a2, a3);	      
	      //Metropolis step
	      U = genunf(0.0, 1.0);
	      if(log(U) < my_fmin(0.0, q))
		{
		  return s2*w/(1.0+w);
		}
	    }	  
	}
    }
  return xkl;
}


void 
update(double *X, double *C, double *b, size_t n)
{
  size_t k, l;
  double xkl, xkl_new;
  for(k=0; k<n; k++)
    {
      for(l=0; l<n; l++)
	{
	  if((C[n*k+l]+C[n*l+k]) > 0)
	    {
	      //Old element at position xkl
	      xkl = X[n*k+l];
	      //New element at position xkl
	      xkl_new = sample_quad(X[n*k+l], X[n*k+k], X[n*l+l],
				    C[n*k+l], C[n*l+k], C[n*k+k], C[n*l+l],
				    b[k], b[l]);
	      //Update entries
	      X[n*k+l] = xkl_new;
	      X[n*k+k] += (xkl - xkl_new);
	      X[n*l+k] = xkl_new;
	      X[n*l+l] += (xkl - xkl_new);      

	      //Old element at position xkl
	      xkl = X[n*k+l];
	      //New element at position xkl
	      xkl_new = sample_quad_rw(X[n*k+l], X[n*k+k], X[n*l+l],
				    C[n*k+l], C[n*l+k], C[n*k+k], C[n*l+l],
				    b[k], b[l]);
	      //Update entries
	      X[n*k+l] = xkl_new;
	      X[n*k+k] += (xkl - xkl_new);
	      X[n*l+k] = xkl_new;
	      X[n*l+l] += (xkl - xkl_new); 	      
	    }
	}
    }
}

void
update_sparse(double *X, double *C, double *b, size_t n, size_t * I, size_t *J, size_t n_idx)
{
  size_t i;
  size_t k, l;
  double xkl, xkl_new;
  for (i=0; i<n_idx; i++)
    {
      k = I[i];
      l = J[i];
      if (k!=l)
	{
	  if((C[n*k+l]+C[n*l+k]) > 0)
	    {
	      //Old element at position xkl
	      xkl = X[n*k+l];
	      //New element at position xkl
	      xkl_new = sample_quad(X[n*k+l], X[n*k+k], X[n*l+l],
				    C[n*k+l], C[n*l+k], C[n*k+k], C[n*l+l],
				    b[k], b[l]);
	      //Update entries
	      X[n*k+l] = xkl_new;
	      X[n*k+k] += (xkl - xkl_new);
	      X[n*l+k] = xkl_new;
	      X[n*l+l] += (xkl - xkl_new);      
	      
	      //Old element at position xkl
	      xkl = X[n*k+l];
	      //New element at position xkl
	      xkl_new = sample_quad_rw(X[n*k+l], X[n*k+k], X[n*l+l],
				       C[n*k+l], C[n*l+k], C[n*k+k], C[n*l+l],
				       b[k], b[l]);
	      //Update entries
	      X[n*k+l] = xkl_new;
	      X[n*k+k] += (xkl - xkl_new);
	      X[n*l+k] = xkl_new;
	      X[n*l+l] += (xkl - xkl_new); 	      
	    }
	}
    }
}
