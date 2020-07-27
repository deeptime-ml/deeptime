# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <string.h>

# include "ranlib.h"
# include "rnglib.h"

/******************************************************************************/

char ch_cap ( char ch )

/******************************************************************************/
/*
  Purpose:

    CH_CAP capitalizes a single character.

  Discussion:

    This routine should be equivalent to the library "toupper" function.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    19 July 1998

  Author:

    John Burkardt

  Parameters:

    Input, char CH, the character to capitalize.

    Output, char CH_CAP, the capitalized character.
*/
{
  if ( 97 <= ch && ch <= 122 )
  {
    ch = ch - 32;
  }

  return ch;
}
/******************************************************************************/

float genbet ( float aa, float bb )

/******************************************************************************/
/*
  Purpose:

    GENBET generates a beta random deviate.

  Discussion:

    This procedure returns a single random deviate from the beta distribution
    with parameters A and B.  The density is

      x^(a-1) * (1-x)^(b-1) / Beta(a,b) for 0 < x < 1

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    19 September 2014

  Author:

    Original FORTRAN77 version by Barry Brown, James Lovato.
    C version by John Burkardt.

  Reference:

    Russell Cheng,
    Generating Beta Variates with Nonintegral Shape Parameters,
    Communications of the ACM,
    Volume 21, Number 4, April 1978, pages 317-322.

  Parameters:

    Input, float AA, the first parameter of the beta distribution.
    0.0 < AA.

    Input, float BB, the second parameter of the beta distribution.
    0.0 < BB.

    Output, float GENBET, a beta random variate.
*/
{
  float a;
  float alpha;
  float b;
  float beta;
  float delta;
  float gamma;
  float k1;
  float k2;
  const float log4 = 1.3862943611198906188;
  const float log5 = 1.6094379124341003746;
  float r;
  float s;
  float t;
  float u1;
  float u2;
  float v;
  float value;
  float w;
  float y;
  float z;

  if ( aa <= 0.0 )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "GENBET - Fatal error!\n" );
    fprintf ( stderr, "  AA <= 0.0\n" );
    exit ( 1 );
  }

  if ( bb <= 0.0 ) 
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "GENBET - Fatal error!\n" );
    fprintf ( stderr, "  BB <= 0.0\n" );
    exit ( 1 );
  }
/*
  Algorithm BB
*/
  if ( 1.0 < aa && 1.0 < bb )
  {
    a = r4_min ( aa, bb );
    b = r4_max ( aa, bb );
    alpha = a + b;
    beta = sqrt ( ( alpha - 2.0 ) / ( 2.0 * a * b - alpha ) );
    gamma = a + 1.0 / beta;

    for ( ; ; )
    {
      u1 = r4_uni_01 ( );
      u2 = r4_uni_01 ( );
      v = beta * log ( u1 / ( 1.0 - u1 ) );
/*
  exp ( v ) replaced by r4_exp ( v )
*/
      w = a * r4_exp ( v );

      z = u1 * u1 * u2;
      r = gamma * v - log4;
      s = a + r - w;

      if ( 5.0 * z <= s + 1.0 + log5 )
      {
        break;
      }

      t = log ( z );
      if ( t <= s )
      {
        break;
      }

      if ( t <= ( r + alpha * log ( alpha / ( b + w ) ) ) )
      {
        break;
      }
    }
  }
/*
  Algorithm BC
*/
  else
  {
    a = r4_max ( aa, bb );
    b = r4_min ( aa, bb );
    alpha = a + b;
    beta = 1.0 / b;
    delta = 1.0 + a - b;
    k1 = delta * ( 1.0 / 72.0 + b / 24.0 ) 
      / ( a / b - 7.0 / 9.0 );
    k2 = 0.25 + ( 0.5 + 0.25 / delta ) * b;

    for ( ; ; )
    {
      u1 = r4_uni_01 ( );
      u2 = r4_uni_01 ( );

      if ( u1 < 0.5 )
      {
        y = u1 * u2;
        z = u1 * y;

        if ( k1 <= 0.25 * u2 + z - y )
        {
          continue;
        }
      }
      else
      {
        z = u1 * u1 * u2;

        if ( z <= 0.25 )
        {
          v = beta * log ( u1 / ( 1.0 - u1 ) );
          w = a * exp ( v );

          if ( aa == a )
          {
            value = w / ( b + w );
          }
          else
          {
            value = b / ( b + w );
          }
          return value;
        }

        if ( k2 < z )
        {
          continue;
        }
      }

      v = beta * log ( u1 / ( 1.0 - u1 ) );
      w = a * exp ( v );

      if ( log ( z ) <= alpha * ( log ( alpha / ( b + w ) ) + v ) - log4 )
      {
        break;
      }
    }
  }

  if ( aa == a )
  {
    value = w / ( b + w );
  }
  else
  {
    value = b / ( b + w );
  }
  return value;
}
/******************************************************************************/

float genchi ( float df )

/******************************************************************************/
/*
  Purpose:

    GENCHI generates a Chi-Square random deviate.

  Discussion:

    This procedure generates a random deviate from the chi square distribution
    with DF degrees of freedom random variable.

    The algorithm exploits the relation between chisquare and gamma.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    01 April 2013

  Author:

    Original FORTRAN77 version by Barry Brown, James Lovato.
    C version by John Burkardt.

  Parameters:

    Input, float DF, the degrees of freedom.
    0.0 < DF.

    Output, float GENCHI, a random deviate from the distribution.
*/
{
  float arg1;
  float arg2;
  float value;

  if ( df <= 0.0 )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "GENCHI - Fatal error!\n" );
    fprintf ( stderr, "  DF <= 0.\n" );
    fprintf ( stderr, "  Value of DF: %g\n", df );
    exit ( 1 );
  }

  arg1 = 1.0;
  arg2 = df / 2.0;

  value = 2.0 * gengam ( arg1, arg2 );

  return value;
}
/******************************************************************************/

float genexp ( float av )

/******************************************************************************/
/*
  Purpose:

    GENEXP generates an exponential random deviate.

  Discussion:

    This procedure generates a single random deviate from an exponential
    distribution with mean AV.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    01 April 2013

  Author:

    Original FORTRAN77 version by Barry Brown, James Lovato.
    C version by John Burkardt.

  Reference:

    Joachim Ahrens, Ulrich Dieter,
    Computer Methods for Sampling From the
    Exponential and Normal Distributions,
    Communications of the ACM,
    Volume 15, Number 10, October 1972, pages 873-882.

  Parameters:

    Input, float AV, the mean of the exponential distribution 
    from which a random deviate is to be generated.

    Output, float GENEXP, a random deviate from the distribution.
*/
{
  float value;

  value = sexpo ( ) * av;

  return value;
}
/******************************************************************************/

float genf ( float dfn, float dfd )

/******************************************************************************/
/*
  Purpose:

    GENF generates an F random deviate.

  Discussion:

    This procedure generates a random deviate from the F (variance ratio)
    distribution with DFN degrees of freedom in the numerator
    and DFD degrees of freedom in the denominator.

    It directly generates the ratio of chisquare variates

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    01 April 2013

  Author:

    Original FORTRAN77 version by Barry Brown, James Lovato.
    C version by John Burkardt.

  Parameters:

    Input, float DFN, the numerator degrees of freedom.
    0.0 < DFN.

    Input, float DFD, the denominator degrees of freedom.
    0.0 < DFD.

    Output, float GENF, a random deviate from the distribution.
*/
{
  float value;
  float xden;
  float xnum;

  if ( dfn <= 0.0 )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "GENF - Fatal error!\n" );
    fprintf ( stderr, "  DFN <= 0.0\n" );
    exit ( 1 );
  }

  if ( dfd <= 0.0 )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "GENF - Fatal error!\n" );
    fprintf ( stderr, "  DFD <= 0.0\n" );
    exit ( 1 );
  }

  xnum = genchi ( dfn ) / dfn;
  xden = genchi ( dfd ) / dfd;
  value = xnum / xden;

  return value;
}
/******************************************************************************/

float gengam ( float a, float r )

/******************************************************************************/
/*
  Purpose:

    GENGAM generates a Gamma random deviate.

  Discussion:

    This procedure generates random deviates from the gamma distribution whose
    density is (A^R)/Gamma(R) * X^(R-1) * Exp(-A*X)

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    01 April 2013

  Author:

    Original FORTRAN77 version by Barry Brown, James Lovato.
    C version by John Burkardt.

  Reference:

    Joachim Ahrens, Ulrich Dieter,
    Generating Gamma Variates by a Modified Rejection Technique,
    Communications of the ACM,
    Volume 25, Number 1, January 1982, pages 47-54.

    Joachim Ahrens, Ulrich Dieter,
    Computer Methods for Sampling from Gamma, Beta, Poisson and
    Binomial Distributions,
    Computing,
    Volume 12, Number 3, September 1974, pages 223-246.

  Parameters:

    Input, float A, the location parameter.

    Input, float R, the shape parameter.

    Output, float GENGAM, a random deviate from the distribution.
*/
{
  float value;

  value = sgamma ( r ) / a;

  return value;
}
/******************************************************************************/

float *genmn ( float parm[] )

/******************************************************************************/
/*
  Purpose:

    GENMN generates a multivariate normal deviate.

  Discussion:

    The method is:
    1) Generate P independent standard normal deviates - Ei ~ N(0,1)
    2) Using Cholesky decomposition find A so that A'*A = COVM
    3) A' * E + MEANV ~ N(MEANV,COVM)

    Note that PARM contains information needed to generate the
    deviates, and is set up by SETGMN.

    PARM(1) contains the size of the deviates, P
    PARM(2:P+1) contains the mean vector.
    PARM(P+2:P*(P+3)/2+1) contains the upper half of the Cholesky
    decomposition of the covariance matrix.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    02 April 2013

  Author:

    Original FORTRAN77 version by Barry Brown, James Lovato.
    C version by John Burkardt.

  Parameters:

    Input, float PARM[P*(P+3)/2+1], parameters set by SETGMN.!

    Output, float GENMN[P], a random deviate from the distribution.
*/
{
  float ae;
  int i;
  int icount;
  int j;
  int p;
  float *work;
  float *x;

  p = ( int ) ( parm[0] );
/*
  Generate P independent normal deviates.
*/
  work = ( float * ) malloc ( p * sizeof ( float ) );

  for ( i = 0; i < p; i++ )
  {
    work[i] = snorm ( );
  }
/*
  Compute X = MEANV + A' * WORK
*/
  x = ( float * ) malloc ( p * sizeof ( float ) );

  for ( i = 0; i < p; i++ )
  {
    icount = 0;
    ae = 0.0;
    for ( j = 0; j <= i; j++ )
    {
      icount = icount + j;
      ae = ae + parm[i+j*p-icount+p+1] * work[j];
    }
    x[i] = ae + parm[i+1];
  }

  free ( work );

  return x;
}
/******************************************************************************/

int *genmul ( int n, float p[], int ncat )

/******************************************************************************/
/*
  Purpose:

    GENMUL generates a multinomial random deviate.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    02 April 2013

  Author:

    Original FORTRAN77 version by Barry Brown, James Lovato.
    C version by John Burkardt.

  Reference:

    Luc Devroye,
    Non-Uniform Random Variate Generation,
    Springer, 1986,
    ISBN: 0387963057,
    LC: QA274.D48.

  Parameters:

    Input, int N, the number of events, which will be
    classified into one of the NCAT categories.

    Input, float P[NCAT-1].  P(I) is the probability that an event
    will be classified into category I.  Thus, each P(I) must be between 
    0.0 and 1.0.  Only the first NCAT-1 values of P must be defined since 
    P(NCAT) would be 1.0 minus the sum of the first NCAT-1 P's.

    Input, int NCAT, the number of categories.

    Output, int GENMUL[NCAT], a random observation from 
    the multinomial distribution.  All IX(i) will be nonnegative and their 
    sum will be N.
*/
{
  int i;
  int icat;
  int *ix;
  int ntot;
  float prob;
  float ptot;

  if ( n < 0 )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "GENMUL - Fatal error!\n" );
    fprintf ( stderr, "  N < 0\n" );
    exit ( 1 );
  }

  if ( ncat <= 1 )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "GENMUL - Fatal error!\n" );
    fprintf ( stderr, "  NCAT <= 1\n" );
    exit ( 1 );
  }

  for ( i = 0; i < ncat - 1; i++ )
  {
    if ( p[i] < 0.0 )
    {
      fprintf ( stderr, "\n" );
      fprintf ( stderr, "GENMUL - Fatal error!\n" );
      fprintf ( stderr, "  Some P(i) < 0.\n" );
      exit ( 1 );
    }

    if ( 1.0 < p[i] )
    {
      fprintf ( stderr, "\n" );
      fprintf ( stderr, "GENMUL - Fatal error!\n" );
      fprintf ( stderr, "  Some 1 < P(i).\n" );
      exit ( 1 );
    }
  }

  ptot = 0.0;
  for ( i = 0; i < ncat - 1; i++ )
  {
    ptot = ptot + p[i];
  }

  if ( 0.99999 < ptot ) 
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "GENMUL - Fatal error!\n" );
    fprintf ( stderr, "  1 < Sum of P().\n" );
    exit ( 1 );
  }
/*
  Initialize variables.
*/
  ntot = n;
  ptot = 1.0;

  ix = ( int * ) malloc ( ncat * sizeof ( int ) );
  for ( i = 0; i < ncat; i++ )
  {
    ix[i] = 0;
  }
/*
  Generate the observation.
*/
  for ( icat = 0; icat < ncat - 1; icat++ )
  {
    prob = p[icat] / ptot;
    ix[icat] = ignbin ( ntot, prob );
    ntot = ntot - ix[icat];
    if ( ntot <= 0 )
    {
      return ix;
    }
    ptot = ptot - p[icat];
  }

  ix[ncat-1] = ntot;

  return ix;
}
/******************************************************************************/

float gennch ( float df, float xnonc )

/******************************************************************************/
/*
  Purpose:

    GENNCH generates a noncentral Chi-Square random deviate.

  Discussion:

    This procedure generates a random deviate from the  distribution of a
    noncentral chisquare with DF degrees of freedom and noncentrality parameter
    XNONC.

    It uses the fact that the noncentral chisquare is the sum of a chisquare
    deviate with DF-1 degrees of freedom plus the square of a normal
    deviate with mean XNONC and standard deviation 1.

    A subtle ambiguity arises in the original formulation:

      gennch = genchi ( arg1 ) + ( gennor ( arg2, arg3 ) ) ** 2

    because the compiler is free to invoke either genchi or gennor
    first, both of which alter the random number generator state,
    resulting in two distinct possible results.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    01 April 2013

  Author:

    Original FORTRAN77 version by Barry Brown, James Lovato.
    C version by John Burkardt.

  Parameters:

    Input, float DF, the degrees of freedom.
    1.0 < DF.

    Input, float XNONC, the noncentrality parameter.
    0.0 <= XNONC.

    Output, float GENNCH, a random deviate from the distribution.
*/
{
  float arg1;
  float arg2;
  float arg3;
  float t1;
  float t2;
  float value;

  if ( df <= 1.0 )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "GENNCH - Fatal error!\n" );
    fprintf ( stderr, "  DF <= 1.\n" );
    exit ( 1 );
  }

  if ( xnonc < 0.0 )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "GENNCH - Fatal error!\n" );
    fprintf ( stderr, "  XNONC < 0.0.\n" );
    exit ( 1 );
  }

  arg1 = df - 1.0;
  arg2 = sqrt ( xnonc );
  arg3 = 1.0;

  t1 = genchi ( arg1 );
  t2 = gennor ( arg2, arg3 );

  value = t1 + t2 * t2;

  return value;
}
/******************************************************************************/

float gennf ( float dfn, float dfd, float xnonc )

/******************************************************************************/
/*
  Purpose:

    GENNF generates a noncentral F random deviate.

  Discussion:

    This procedure generates a random deviate from the noncentral F
    (variance ratio) distribution with DFN degrees of freedom in the
    numerator, and DFD degrees of freedom in the denominator, and
    noncentrality parameter XNONC.

    It directly generates the ratio of noncentral numerator chisquare variate
    to central denominator chisquare variate.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    01 April 2013

  Author:

    Original FORTRAN77 version by Barry Brown, James Lovato.
    C version by John Burkardt.

  Parameters:

    Input, float DFN, the numerator degrees of freedom.
    1.0 < DFN.

    Input, float DFD, the denominator degrees of freedom.
    0.0 < DFD.

    Input, float XNONC, the noncentrality parameter.
    0.0 <= XNONC.

    Output, float GENNF, a random deviate from the distribution.
*/
{
  float value;
  float xden;
  float xnum;

  if ( dfn <= 1.0 )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "GENNF - Fatal error!\n" );
    fprintf ( stderr, "  DFN <= 1.0\n" );
    exit ( 1 );
  }

  if ( dfd <= 0.0 )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "GENNF - Fatal error!\n" );
    fprintf ( stderr, "  DFD <= 0.0\n" );
    exit ( 1 );
  }

  if ( xnonc < 0.0 )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "GENNF - Fatal error!\n" );
    fprintf ( stderr, "  XNONC < 0.0\n" );
    exit ( 1 );
  }

  xnum = gennch ( dfn, xnonc ) / dfn;
  xden = genchi ( dfd ) / dfd;

  value = xnum / xden;

  return value;
}
/******************************************************************************/

float gennor ( float av, float sd )

/******************************************************************************/
/*
  Purpose:

    GENNOR generates a normal random deviate.

  Discussion:

    This procedure generates a single random deviate from a normal distribution
    with mean AV, and standard deviation SD.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    01 April 2013

  Author:

    Original FORTRAN77 version by Barry Brown, James Lovato.
    C version by John Burkardt.

  Reference:

    Joachim Ahrens, Ulrich Dieter,
    Extensions of Forsythe's Method for Random
    Sampling from the Normal Distribution,
    Mathematics of Computation,
    Volume 27, Number 124, October 1973, page 927-937.

  Parameters:

    Input, float AV, the mean.

    Input, float SD, the standard deviation.

    Output, float GENNOR, a random deviate from the distribution.
*/
{
  float value;

  value = sd * snorm ( ) + av;

  return value;
}
/******************************************************************************/

void genprm ( int iarray[], int n )

/******************************************************************************/
/*
  Purpose:

    GENPRM generates and applies a random permutation to an array.

  Discussion:

    To see the permutation explicitly, let the input array be
    1, 2, ..., N.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    01 April 2013

  Author:

    Original FORTRAN77 version by Barry Brown, James Lovato.
    C version by John Burkardt.

  Parameters:

    Input/output, int IARRAY(N), an array to be permuted.

    Input, int N, the number of entries in the array.
*/
{
  int i;
  int itmp;
  int iwhich;

  for ( i = 1; i <= n; i++ )
  {
    iwhich = ignuin ( i, n );
    itmp = iarray[iwhich-1];
    iarray[iwhich-1] = iarray[i-1];
    iarray[i-1] = itmp;
  }
  return;
}
/******************************************************************************/

float genunf ( float low, float high )

/******************************************************************************/
/*
  Purpose:

    GENUNF generates a uniform random deviate.

  Discussion:

    This procedure generates a real deviate uniformly distributed between
    LOW and HIGH.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    01 April 2013

  Author:

    Original FORTRAN77 version by Barry Brown, James Lovato.
    C version by John Burkardt.

  Parameters:

    Input, float LOW, HIGH, the lower and upper bounds.

    Output, float GENUNF, a random deviate from the distribution.
*/
{
  float value;

  value = low + ( high - low ) * r4_uni_01 ( );

  return value;
}
/******************************************************************************/

int i4_max ( int i1, int i2 )

/******************************************************************************/
/*
  Purpose:

    I4_MAX returns the maximum of two I4's.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    29 August 2006

  Author:

    John Burkardt

  Parameters:

    Input, int I1, I2, are two integers to be compared.

    Output, int I4_MAX, the larger of I1 and I2.
*/
{
  int value;

  if ( i2 < i1 )
  {
    value = i1;
  }
  else
  {
    value = i2;
  }
  return value;
}
/******************************************************************************/

int i4_min ( int i1, int i2 )

/******************************************************************************/
/*
  Purpose:

    I4_MIN returns the smaller of two I4's.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    29 August 2006

  Author:

    John Burkardt

  Parameters:

    Input, int I1, I2, two integers to be compared.

    Output, int I4_MIN, the smaller of I1 and I2.
*/
{
  int value;

  if ( i1 < i2 )
  {
    value = i1;
  }
  else
  {
    value = i2;
  }
  return value;
}
/******************************************************************************/

int ignbin ( int n, float pp )

/******************************************************************************/
/*
  Purpose:

    IGNBIN generates a binomial random deviate.

  Discussion:

    This procedure generates a single random deviate from a binomial
    distribution whose number of trials is N and whose
    probability of an event in each trial is P.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    01 April 2013

  Author:

    Original FORTRAN77 version by Barry Brown, James Lovato.
    C version by John Burkardt.

  Reference:

    Voratas Kachitvichyanukul, Bruce Schmeiser,
    Binomial Random Variate Generation,
    Communications of the ACM,
    Volume 31, Number 2, February 1988, pages 216-222.

  Parameters:

    Input, int N, the number of binomial trials, from which a
    random deviate will be generated.
    0 < N.

    Input, float PP, the probability of an event in each trial of
    the binomial distribution from which a random deviate is to be generated.
    0.0 < PP < 1.0.

    Output, int IGNBIN, a random deviate from the
    distribution.
*/
{
  float al;
  float alv;
  float amaxp;
  float c;
  float f;
  float f1;
  float f2;
  float ffm;
  float fm;
  float g;
  int i;
  int ix;
  int ix1;
  int k;
  int m;
  int mp;
  float p;
  float p1;
  float p2;
  float p3;
  float p4;
  float q;
  float qn;
  float r;
  float t;
  float u;
  float v;
  int value;
  float w;
  float w2;
  float x;
  float x1;
  float x2;
  float xl;
  float xll;
  float xlr;
  float xm;
  float xnp;
  float xnpq;
  float xr;
  float ynorm;
  float z;
  float z2;

  if ( pp <= 0.0 || 1.0 <= pp )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "IGNBIN - Fatal error!\n" );
    fprintf ( stderr, "  PP is out of range.\n" );
    exit ( 1 );
  }

  p = r4_min ( pp, 1.0 - pp );
  q = 1.0 - p;
  xnp = ( float ) ( n ) * p;

  if ( xnp < 30.0 )
  {
    qn = pow ( q, n );
    r = p / q;
    g = r * ( float ) ( n + 1 );

    for ( ; ; )
    {
      ix = 0;
      f = qn;
      u = r4_uni_01 ( );

      for ( ; ; )
      {
        if ( u < f )
        {
          if ( 0.5 < pp )
          {
            ix = n - ix;
          }
          value = ix;
          return value;
        }

        if ( 110 < ix )
        {
          break;
        }
        u = u - f;
        ix = ix + 1;
        f = f * ( g / ( float ) ( ix ) - r );
      }
    }
  }
  ffm = xnp + p;
  m = ffm;
  fm = m;
  xnpq = xnp * q;
  p1 = ( int ) ( 2.195 * sqrt ( xnpq ) - 4.6 * q ) + 0.5;
  xm = fm + 0.5;
  xl = xm - p1;
  xr = xm + p1;
  c = 0.134 + 20.5 / ( 15.3 + fm );
  al = ( ffm - xl ) / ( ffm - xl * p );
  xll = al * ( 1.0 + 0.5 * al );
  al = ( xr - ffm ) / ( xr * q );
  xlr = al * ( 1.0 + 0.5 * al );
  p2 = p1 * ( 1.0 + c + c );
  p3 = p2 + c / xll;
  p4 = p3 + c / xlr;
/*
  Generate a variate.
*/
  for ( ; ; )
  {
    u = r4_uni_01 ( ) * p4;
    v = r4_uni_01 ( );
/*
  Triangle
*/
    if ( u < p1 )
    {
      ix = xm - p1 * v + u;
      if ( 0.5 < pp ) 
      {
        ix = n - ix;
      }
      value = ix;
      return value;
    }
/*
  Parallelogram
*/
    if ( u <= p2 )
    {
      x = xl + ( u - p1 ) / c;
      v = v * c + 1.0 - fabs ( xm - x ) / p1;

      if ( v <= 0.0 || 1.0 < v )
      {
        continue;
      }
      ix = x;
    }
    else if ( u <= p3 )
    {
      ix = xl + log ( v ) / xll;
      if ( ix < 0 )
      {
        continue;
      }
      v = v * ( u - p2 ) * xll;
    }
    else
    {
      ix = xr - log ( v ) / xlr;
      if ( n < ix )
      {
        continue;
      }
      v = v * ( u - p3 ) * xlr;
    }
    k = abs ( ix - m );

    if ( k <= 20 || xnpq / 2.0 - 1.0 <= k )
    {
      f = 1.0;
      r = p / q;
      g = ( n + 1 ) * r;

      if ( m < ix )
      {
        mp = m + 1;
        for ( i = mp; i <= ix; i++ )
        {
          f = f * ( g / i - r );
        }
      }
      else if ( ix < m )
      {
        ix1 = ix + 1;
        for ( i = ix1; i <= m; i++ )
        {
          f = f / ( g / i - r );
        }
      }

      if ( v <= f )
      {
        if ( 0.5 < pp )
        {
          ix = n - ix;
        }
        value = ix;
        return value;
      }
    }
    else
    {
      amaxp = ( k / xnpq ) * ( ( k * ( k / 3.0 
        + 0.625 ) + 0.1666666666666 ) / xnpq + 0.5 );
      ynorm = - ( float ) ( k * k ) / ( 2.0 * xnpq );
      alv = log ( v );

      if ( alv < ynorm - amaxp )
      {
        if ( 0.5 < pp )
        {
          ix = n - ix;
        }
        value = ix;
        return value;
      }

      if ( ynorm + amaxp < alv )
      {
        continue;
      }

      x1 = ( float ) ( ix + 1 );
      f1 = fm + 1.0;
      z = ( float ) ( n + 1 ) - fm;
      w = ( float ) ( n - ix + 1 );
      z2 = z * z;
      x2 = x1 * x1;
      f2 = f1 * f1;
      w2 = w * w;

      t = xm * log ( f1 / x1 ) + ( n - m + 0.5 ) * log ( z / w ) 
        + ( float ) ( ix - m ) * log ( w * p / ( x1 * q )) 
        + ( 13860.0 - ( 462.0 - ( 132.0 - ( 99.0 - 140.0 
        / f2 ) / f2 ) / f2 ) / f2 ) / f1 / 166320.0 
        + ( 13860.0 - ( 462.0 - ( 132.0 - ( 99.0 - 140.0 
        / z2 ) / z2 ) / z2 ) / z2 ) / z / 166320.0 
        + ( 13860.0 - ( 462.0 - ( 132.0 - ( 99.0 - 140.0 
        / x2 ) / x2 ) / x2 ) / x2 ) / x1 / 166320.0 
        + ( 13860.0 - ( 462.0 - ( 132.0 - ( 99.0 - 140.0 
        / w2 ) / w2 ) / w2 ) / w2 ) / w / 166320.0;

      if ( alv <= t )
      {
        if ( 0.5 < pp )
        {
          ix = n - ix;
        }
        value = ix;
        return value;
      }
    }
  }
  return value;
}
/******************************************************************************/

int ignnbn ( int n, float p )

/******************************************************************************/
/*
  Purpose:

    IGNNBN generates a negative binomial random deviate.

  Discussion:

    This procedure generates a single random deviate from a negative binomial
    distribution.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    01 April 2013

  Author:

    Original FORTRAN77 version by Barry Brown, James Lovato.
    C version by John Burkardt.

  Reference:

    Luc Devroye,
    Non-Uniform Random Variate Generation,
    Springer, 1986,
    ISBN: 0387963057,
    LC: QA274.D48.

  Parameters:

    Input, int N, the required number of events.
    0 <= N.

    Input, float P, the probability of an event during a 
    Bernoulli trial.  0.0 < P < 1.0.

    Output, int IGNNBN, a random deviate from 
    the distribution.
*/
 {
  float a;
  float r;
  int value;
  float y;

  if ( n < 0 )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "IGNNBN - Fatal error!\n" );
    fprintf ( stderr, "  N < 0.\n" );
    exit ( 1 );
  }

  if ( p <= 0.0 )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "IGNNBN - Fatal error!\n" );
    fprintf ( stderr, "  P <= 0.0\n" );
    exit ( 1 );
  }

  if ( 1.0 <= p )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "IGNNBN - Fatal error!\n" );
    fprintf ( stderr, "  1.0 <= P\n" );
    exit ( 1 );
  }
/*
  Generate Y, a random gamma (n,(1-p)/p) variable.
*/
  r = ( float ) ( n );
  a = p / ( 1.0 - p );
  y = gengam ( a, r );
/*
  Generate a random Poisson ( y ) variable.
*/
  value = ignpoi ( y );

  return value;
}
/******************************************************************************/

int ignpoi ( float mu )

/******************************************************************************/
/*
  Purpose:

    IGNPOI generates a Poisson random deviate.

  Discussion:

    This procedure generates a single random deviate from a Poisson
    distribution with given mean.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    01 April 2013

  Author:

    Original FORTRAN77 version by Barry Brown, James Lovato.
    C version by John Burkardt.

  Reference:

    Joachim Ahrens, Ulrich Dieter,
    Computer Generation of Poisson Deviates
    From Modified Normal Distributions,
    ACM Transactions on Mathematical Software,
    Volume 8, Number 2, June 1982, pages 163-179.

  Parameters:

    Input, float MU, the mean of the Poisson distribution 
    from which a random deviate is to be generated.

    Output, int IGNPOI, a random deviate from
    the distribution.
*/
{
  const float a0 = -0.5;
  const float a1 =  0.3333333;
  const float a2 = -0.2500068;
  const float a3 =  0.2000118;
  const float a4 = -0.1661269;
  const float a5 =  0.1421878;
  const float a6 = -0.1384794;
  const float a7 =  0.1250060;
  float b1;
  float b2;
  float c;
  float c0;
  float c1;
  float c2;
  float c3;
  float d;
  float del;
  float difmuk;
  float e;
  const float fact[10] = { 1.0, 1.0, 2.0, 6.0, 24.0, 
    120.0, 720.0, 5040.0, 40320.0, 362880.0 };
  float fk;
  float fx;
  float fy;
  float g;
  int k;
  int kflag;
  int l;
  int m;
  float omega;
  float p;
  float p0;
  float px;
  float py;
  float q;
  float s;
  float t;
  float u;
  float v;
  int value;
  float x;
  float xx;
/*
  Start new table and calculate P0.
*/
  if ( mu < 10.0 )
  {
    m = i4_max ( 1, ( int ) ( mu ) );
    p = exp ( - mu );
    q = p;
    p0 = p;
/*
  Uniform sample for inversion method.
*/
    for ( ; ; )
    {
      u = r4_uni_01 ( );
      value = 0;

      if ( u <= p0 )
      {
        return value;
      }
/*
  Creation of new Poisson probabilities.
*/
      for ( k = 1; k <= 35; k++ )
      {
        p = p * mu / ( float ) ( k );
        q = q + p;
        if ( u <= q )
        {
          value = k;
          return value;
        }
      }
    }
  }
  else
  {
    s = sqrt ( mu );
    d = 6.0 * mu * mu;
    l = ( int ) ( mu - 1.1484 );
/*
  Normal sample.
*/
    g = mu + s * snorm ( );

    if ( 0.0 <= g )
    {
      value = ( int ) ( g );
/*
  Immediate acceptance if large enough.
*/
      if ( l <= value )
      {
        return value;
      }
/*
  Squeeze acceptance.
*/
      fk = ( float ) ( value );
      difmuk = mu - fk;
      u = r4_uni_01 ( );

      if ( difmuk * difmuk * difmuk <= d * u )
      {
        return value;
      }
    }
/*
  Preparation for steps P and Q.
*/
    omega = 0.3989423 / s;
    b1 = 0.04166667 / mu;
    b2 = 0.3 * b1 * b1;
    c3 = 0.1428571 * b1 * b2;
    c2 = b2 - 15.0 * c3;
    c1 = b1 - 6.0 * b2 + 45.0 * c3;
    c0 = 1.0 - b1 + 3.0 * b2 - 15.0 * c3;
    c = 0.1069 / mu;

    if ( 0.0 <= g )
    {
      kflag = 0;

      if ( value < 10 )
      {
        px = -mu;
        py = pow ( mu, value ) / fact[value];
      }
      else
      {
        del = 0.8333333E-01 / fk;
        del = del - 4.8 * del * del * del;
        v = difmuk / fk;

        if ( 0.25 < fabs ( v ) )
        {
          px = fk * log ( 1.0 + v ) - difmuk - del;
        }
        else
        {
          px = fk * v * v * ((((((( a7 
            * v + a6 ) 
            * v + a5 ) 
            * v + a4 ) 
            * v + a3 ) 
            * v + a2 ) 
            * v + a1 ) 
            * v + a0 ) - del;
        }
        py = 0.3989423 / sqrt ( fk );
      }
      x = ( 0.5 - difmuk ) / s;
      xx = x * x;
      fx = -0.5 * xx;
      fy = omega * ((( c3 * xx + c2 ) * xx + c1 ) * xx + c0 );

      if ( kflag <= 0 )
      {
        if ( fy - u * fy <= py * exp ( px - fx ) )
        {
          return value;
        }
      }
      else
      {
        if ( c * fabs ( u ) <= py * exp ( px + e ) - fy * exp ( fx + e ) )
        {
          return value;
        }
      }
    }
/*
  Exponential sample.
*/
    for ( ; ; )
    {
      e = sexpo ( );
      u = 2.0 * r4_uni_01 ( ) - 1.0;
      if ( u < 0.0 )
      {
        t = 1.8 - fabs ( e );
      }
      else
      {
        t = 1.8 + fabs ( e );
      }

      if ( t <= -0.6744 )
      {
        continue;
      }

      value = ( int ) ( mu + s * t );
      fk = ( float ) ( value );
      difmuk = mu - fk;

      kflag = 1;
/*
  Calculation of PX, PY, FX, FY.
*/
      if ( value < 10 )
      {
        px = -mu;
        py = pow ( mu, value ) / fact[value];
      }
      else
      {
        del = 0.8333333E-01 / fk;
        del = del - 4.8 * del * del * del;
        v = difmuk / fk;

        if ( 0.25 < fabs ( v ) )
        {
          px = fk * log ( 1.0 + v ) - difmuk - del;
        }
        else
        {
          px = fk * v * v * ((((((( a7 
            * v + a6 ) 
            * v + a5 ) 
            * v + a4 ) 
            * v + a3 ) 
            * v + a2 ) 
            * v + a1 ) 
            * v + a0 ) - del;
        }
        py = 0.3989423 / sqrt ( fk );
      }

      x = ( 0.5 - difmuk ) / s;
      xx = x * x;
      fx = -0.5 * xx;
      fy = omega * ((( c3 * xx + c2 ) * xx + c1 ) * xx + c0 );

      if ( kflag <= 0 )
      {
        if ( fy - u * fy <= py * exp ( px - fx ) )
        {
          return value;
        }
      }
      else
      {
        if ( c * fabs ( u ) <= py * exp ( px + e ) - fy * exp ( fx + e ) )
        {
          return value;
        }
      }
    }
  }
}
/******************************************************************************/

int ignuin ( int low, int high )

/******************************************************************************/
/*
  Purpose:

    IGNUIN generates a random integer in a given range.

  Discussion:

    Each deviate K satisfies LOW <= K <= HIGH.

    If (HIGH-LOW) > 2,147,483,561, this procedure prints an error message
    and stops the program.

    IGNLGI generates ints between 1 and 2147483562.

    MAXNUM is 1 less than the maximum generatable value.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    01 April 2013

  Author:

    Original FORTRAN77 version by Barry Brown, James Lovato.
    C version by John Burkardt.

  Parameters:

    Input, int LOW, HIGH, the lower and upper bounds.

    Output, int IGNUIN, a random deviate from 
    the distribution.
*/
{
  int ign;
  int maxnow;
  const int maxnum = 2147483561;
  int ranp1;
  int value;
  int width;

  if ( high < low )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "IGNUIN - Fatal error!\n" );
    fprintf ( stderr, "  HIGH < LOW.\n" );
    exit ( 1 );
  }

  width = high - low;

  if ( maxnum < width )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "IGNUIN - Fatal error!\n" );
    fprintf ( stderr, "  Range HIGH-LOW is too large.\n" );
    exit ( 1 );
  }

  if ( low == high )
  {
    value = low;
    return value;
  }

  ranp1 = width + 1;
  maxnow = ( maxnum / ranp1 ) * ranp1;

  for ( ; ; )
  {
    ign = i4_uni ( ) - 1;

    if ( ign <= maxnow )
    {
      break;
    }
  }

  value = low + ( ign % ranp1 );

  return value;
}
/******************************************************************************/

int lennob ( char *s )

/******************************************************************************/
/*
  Purpose:

    LENNOB counts the length of a string, ignoring trailing blanks.

  Discussion:

    This procedure returns the length of a string up to and including
    the last non-blank character.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    02 April 2013

  Author:

    Original FORTRAN77 version by Barry Brown, James Lovato.
    C version by John Burkardt.

  Parameters:

    Input, char *S, the string.

    Output, int LENNOB, the length of the string to the last
    nonblank.
*/
{
  int n;
  char *t;

  n = strlen ( s );
  t = s + strlen ( s ) - 1;

  while ( 0 < n )
  {
    if ( *t != ' ' )
    {
      return n;
    }
    t--;
    n--;
  }

  return n;
}
/******************************************************************************/

void phrtsd ( char *phrase, int *seed1, int *seed2 )

/******************************************************************************/
/*
  Purpose:

    PHRTST converts a phrase to a pair of random number generator seeds.

  Discussion:

    This procedure uses a character string to generate two seeds for the RGN
    random number generator.

    Trailing blanks are eliminated before the seeds are generated.

    Generated seed values will fall in the range 1 to 2^30 = 1,073,741,824.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    02 April 2013

  Author:

    Original FORTRAN77 version by Barry Brown, James Lovato.
    C version by John Burkardt.

  Parameters:

    Input, char *PHRASE, a phrase to be used for the
    random number generation.

    Output, int *SEED1, *SEED2, the two seeds for the
    random number generator, based on PHRASE.
*/
{
  char c;
  char *cstar;
  int i;
  int ichr;
  int j;
  int lphr;
  int shift[5] = { 1, 64, 4096, 262144, 16777216 };
  char table[] = 
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()_+[];:'""<>?,./";
  int twop30 = 1073741824;
  int values[5];

  *seed1 = 1234567890;
  *seed2 = 123456789;

  lphr = lennob ( phrase );

  for ( i = 0; i < lphr; i++ )
  {
    c = phrase[i];
/*
  CSTAR points to the address of C in TABLE.
*/
    cstar = strchr ( table, c );
/*
  If C does not occur, set the index to 62.
*/
    if ( !cstar )
    {
      ichr = 63;
    }
/*
  Otherwise, the numerical index is the difference between the
  addresses CSTAR and TABLE.
*/
    else
    {
      ichr = cstar - table + 1;
      ichr = ichr % 64;
      if ( ichr == 0 )
      {
        ichr = 63;
      }
    }

    ichr = ichr - 1;

    for ( j = 0; j < 5; j++ )
    {
      values[j] = ichr - j;
      if ( values[j] < 1 )
      {
        values[j] = values[j] + 63;
      }
    }

    for ( j = 0; j < 5; j++ )
    {
      *seed1 = ( *seed1 + shift[j] * values[j]   ) % twop30;
      *seed2 = ( *seed2 + shift[j] * values[4-j] ) % twop30;
    }
  }
  return;
}
/******************************************************************************/

void prcomp ( int maxobs, int p, float mean[], float xcovar[], float answer[] )

/******************************************************************************/
/*
  Purpose:

    PRCOMP prints covariance information.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    02 April 2013

  Author:

    Original FORTRAN77 version by Barry Brown, James Lovato.
    C version by John Burkardt.

  Parameters:

    Input, int P, the number of variables.

    Input, float MEAN[P], the mean for each column.

    Input, float XCOVAR[P*P], the variance/covariance matrix.

    Input, float ANSWER[MAXOBS*P], the observed values.
*/
{
  float dum1;
  float dum2;
  int i;
  int j;
  float *rcovar;
  float *rmean;
  float *rvar;

  printf ( "\n" );
  printf ( "PRCOMP:\n" );
  printf ( "  Print and compare covariance information\n" );
  printf ( "\n" );

  rmean = ( float * ) malloc ( p * sizeof ( float ) );
  rvar = ( float * ) malloc ( p * sizeof ( float ) );

  for ( j = 0; j < p; j++ )
  {
    stats ( answer+j*maxobs, maxobs, rmean+j, rvar+j, &dum1, &dum2 );
    printf ( "  Variable number %d\n", j );
    printf ( "  Mean       %14.6g  Generated %14.6g\n", mean[j], rmean[j] );
    printf ( "  Variance   %14.6g  Generated %14.6g\n", xcovar[j+j*p], rvar[j] );
  }

  printf ( "\n" );
  printf ( "  Covariances:\n" );
  printf ( "\n" );

  rcovar = ( float * ) malloc ( p * p * sizeof ( float ) );

  for ( i = 0; i < p; i++ )
  {
    for ( j = 0; j < i; j++ )
    {
      printf ( "  I = %d, J = %d\n", i, j );
      rcovar[i+j*p] = r4vec_covar ( maxobs, answer+i*p, answer+j*p );
      printf ( "  Covariance %14.6g  Generated %14.6g\n",
        xcovar[i+j*p], rcovar[i+j*p] );
    }
  }

  free ( rcovar );
  free ( rmean );
  free ( rvar );

  return;
}
/******************************************************************************/

float r4_exp ( float x )

/******************************************************************************/
/*
  Purpose:

    R4_EXP computes the exponential function, avoiding overflow and underflow.

  Discussion:

    For arguments of very large magnitude, the evaluation of the
    exponential function can cause computational problems.  Some languages
    and compilers may return an infinite value or a "Not-a-Number".  
    An alternative, when dealing with a wide range of inputs, is simply
    to truncate the calculation for arguments whose magnitude is too large.
    Whether this is the right or convenient approach depends on the problem
    you are dealing with, and whether or not you really need accurate
    results for large magnitude inputs, or you just want your code to
    stop crashing.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    19 September 2014

  Author:

    John Burkardt

  Parameters:

    Input, float X, the argument of the exponential function.

    Output, float R4_EXP, the value of exp ( X ).
*/
{
  const float r4_huge = 1.0E+30;
  const float r4_log_max = +69.0776;
  const float r4_log_min = -69.0776;
  float value;

  if ( x <= r4_log_min )
  {
    value = 0.0;
  }
  else if ( x < r4_log_max )
  {
    value = exp ( x );
  }
  else
  {
    value = r4_huge;
  }

  return value;
}
/******************************************************************************/

float r4_exponential_sample ( float lambda )

/******************************************************************************/
/*
  Purpose:

    R4_EXPONENTIAL_SAMPLE samples the exponential PDF.

  Discussion:

    Note that the parameter LAMBDA is a multiplier.  In some formulations,
    it is used as a divisor instead.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    18 April 2013

  Author:

    John Burkardt

  Parameters:

    Input, float LAMBDA, the parameter of the PDF.

    Output, float R4_EXPONENTIAL_SAMPLE, a sample of the PDF.
*/
{
  float r;
  float value;

  r = r4_uni_01 ( );

  value = - log ( r ) * lambda;

  return value;
}
/******************************************************************************/

float r4_max ( float x, float y )

/******************************************************************************/
/*
  Purpose:

    R4_MAX returns the maximum of two R4's.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    07 May 2006

  Author:

    John Burkardt

  Parameters:

    Input, float X, Y, the quantities to compare.

    Output, float R4_MAX, the maximum of X and Y.
*/
{
  float value;

  if ( y < x )
  {
    value = x;
  }
  else
  {
    value = y;
  }
  return value;
}
/******************************************************************************/

float r4_min ( float x, float y )

/******************************************************************************/
/*
  Purpose:

    R4_MIN returns the minimum of two R4's.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    07 May 2006

  Author:

    John Burkardt

  Parameters:

    Input, float X, Y, the quantities to compare.

    Output, float R4_MIN, the minimum of X and Y.
*/
{
  float value;

  if ( y < x )
  {
    value = y;
  }
  else
  {
    value = x;
  }
  return value;
}
/******************************************************************************/

float r4vec_covar ( int n, float x[], float y[] )

/******************************************************************************/
/*
  Purpose:

    R4VEC_COVAR computes the covariance of two vectors.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    20 April 2013

  Author:

    John Burkardt.

  Parameters:

    Input, float X[N], Y[N], the two vectors.

    Input, int N, the dimension of the two vectors.

    Output, float R4VEC_COVAR, the covariance of the two vectors.
*/
{
  int i;
  float value;
  float x_average;
  float y_average;

  x_average = 0.0;
  for ( i = 0; i < n; i++ )
  {
    x_average = x_average + x[i];
  }
  x_average = x_average / ( float ) ( n );

  y_average = 0.0;
  for ( i = 0; i < n; i++ )
  {
    y_average = y_average + y[i];
  }
  y_average = y_average / ( float ) ( n );

  value = 0.0;
  for ( i = 0; i < n; i++ )
  {
    value = value + ( x[i] - x_average ) * ( y[i] - y_average );
  }

  value = value / ( float ) ( n - 1 );

  return value;
}
/******************************************************************************/

int s_eqi ( char *s1, char *s2 )

/******************************************************************************/
/*
  Purpose:

    S_EQI reports whether two strings are equal, ignoring case.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    03 June 2008

  Author:

    John Burkardt

  Parameters:

    Input, char *S1, char *S2, pointers to two strings.

    Output, int S_EQI, is true if the strings are equal.
*/
{
  int i;
  int nchar;
  int nchar1;
  int nchar2;

  nchar1 = strlen ( s1 );
  nchar2 = strlen ( s2 );
  if ( nchar1 < nchar2 )
  {
    nchar = nchar1;
  }
  else
  {
    nchar = nchar2;
  }
/*
  The strings are not equal if they differ over their common length.
*/
  for ( i = 0; i < nchar; i++ )
  {

    if ( ch_cap ( s1[i] ) != ch_cap ( s2[i] ) )
    {
      return 0;
    }
  }
/*
  The strings are not equal if the longer one includes nonblanks
  in the tail.
*/
  if ( nchar < nchar1 )
  {
    for ( i = nchar; i < nchar1; i++ )
    {
      if ( s1[i] != ' ' )
      {
        return 0;
      }
    }
  }
  else if ( nchar < nchar2 )
  {
    for ( i = nchar; i < nchar2; i++ )
    {
      if ( s2[i] != ' ' )
      {
        return 0;
      }
    }
  }

  return 1;
}
/******************************************************************************/

float sdot ( int n, float dx[], int incx, float dy[], int incy )

/******************************************************************************/
/*
  Purpose:

    SDOT forms the dot product of two vectors.

  Discussion:

    This routine uses unrolled loops for increments equal to one.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    29 March 2007

  Author:

    C version by John Burkardt

  Reference:

    Jack Dongarra, Cleve Moler, Jim Bunch, Pete Stewart,
    LINPACK User's Guide,
    SIAM, 1979,
    ISBN13: 978-0-898711-72-1,
    LC: QA214.L56.

    Charles Lawson, Richard Hanson, David Kincaid, Fred Krogh,
    Algorithm 539: 
    Basic Linear Algebra Subprograms for Fortran Usage,
    ACM Transactions on Mathematical Software, 
    Volume 5, Number 3, September 1979, pages 308-323.

  Parameters:

    Input, int N, the number of entries in the vectors.

    Input, float DX[*], the first vector.

    Input, int INCX, the increment between successive entries in DX.

    Input, float DY[*], the second vector.

    Input, int INCY, the increment between successive entries in DY.

    Output, float SDOT, the sum of the product of the corresponding
    entries of DX and DY.
*/
{
  float dtemp;
  int i;
  int ix;
  int iy;
  int m;

  dtemp = 0.0;

  if ( n <= 0 )
  {
    return dtemp;
  }
/*
  Code for unequal increments or equal increments
  not equal to 1.
*/
  if ( incx != 1 || incy != 1 )
  {
    if ( 0 <= incx )
    {
      ix = 0;
    }
    else
    {
      ix = ( - n + 1 ) * incx;
    }

    if ( 0 <= incy )
    {
      iy = 0;
    }
    else
    {
      iy = ( - n + 1 ) * incy;
    }

    for ( i = 0; i < n; i++ )
    {
      dtemp = dtemp + dx[ix] * dy[iy];
      ix = ix + incx;
      iy = iy + incy;
    }
  }
/*
  Code for both increments equal to 1.
*/
  else
  {
    m = n % 5;

    for ( i = 0; i < m; i++ )
    {
      dtemp = dtemp + dx[i] * dy[i];
    }

    for ( i = m; i < n; i = i + 5 )
    {
      dtemp = dtemp + dx[i  ] * dy[i  ] 
                    + dx[i+1] * dy[i+1] 
                    + dx[i+2] * dy[i+2] 
                    + dx[i+3] * dy[i+3] 
                    + dx[i+4] * dy[i+4];
    }

  }

  return dtemp;
}
/******************************************************************************/

float *setcov ( int p, float var[], float corr )

/******************************************************************************/
/*
  Purpose:

    SETCOV sets a covariance matrix from variance and common correlation.

  Discussion:

    This procedure sets the covariance matrix from the variance and
    common correlation.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    02 April 2013

  Author:

    Original FORTRAN77 version by Barry Brown, James Lovato.
    C version by John Burkardt.

  Parameters:

    Input, int P, the number of variables.

    Input, float VAR[P], the variances.

    Input, float CORR, the common correlaton.

    Output, float COVAR[P*P], the covariance matrix.
*/
{
  float *covar;
  int i;
  int j;

  covar = ( float * ) malloc ( p * p * sizeof ( float ) );

  for ( j = 0; j < p; j++ )
  {
    for ( i = 0; i < p; i++ )
    {
      if ( i == j )
      {
        covar[i+j*p] = var[i];
      }
      else
      {
        covar[i+j*p] = corr * sqrt ( var[i] * var[j] );
      }
    }
  }

  return covar;
}
/******************************************************************************/

void setgmn ( float meanv[], float covm[], int p, float parm[] )

/******************************************************************************/
/*
  Purpose:

    SETGMN sets data for the generation of multivariate normal deviates.

  Discussion:

    This procedure places P, MEANV, and the Cholesky factorization of
    COVM in GENMN.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    02 April 2013

  Author:

    Original FORTRAN77 version by Barry Brown, James Lovato.
    C version by John Burkardt.

  Parameters:

    Input, float MEANV[P], the means of the multivariate 
    normal distribution.

    Input/output, float COVM[P*P].  On input, the covariance
    matrix of the multivariate distribution.  On output, the information 
    in COVM has been overwritten.

    Input, int P, the number of dimensions.

    Output, float PARM[P*(P+3)/2+1], parameters needed to generate
    multivariate normal deviates.
*/
{
  int i;
  int icount;
  int info;
  int j;

  if ( p <= 0 )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "SETGMN - Fatal error!\n" );
    fprintf ( stderr, "  P was not positive.\n" );
    exit ( 1 );
  }
/*
  Store P.
*/
  parm[0] = p;
/*
  Store MEANV.
*/
  for ( i = 0; i < p; i++ )
  {
    parm[1+i] = meanv[i];
  }
/*
  Compute the Cholesky decomposition.
*/
  info = spofa ( covm, p, p );

  if ( info != 0 )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "SETGMN - Fatal error!\n" );
    fprintf ( stderr, "  SPOFA finds COVM not positive definite.\n" );
    exit ( 1 );
  }
/*
  Store the upper half of the Cholesky factor.
*/
  icount = p + 1;

  for ( i = 0; i < p; i++ )
  {
    for ( j = i; j < p; j++ )
    {
      parm[icount] = covm[i+j*p];
      icount = icount + 1;
    }
  }
  return;
}
/******************************************************************************/

float sexpo (void)

/******************************************************************************/
/*
  Purpose:

    SEXPO samples the standard exponential distribution.

  Discussion:

    This procedure corresponds to algorithm SA in the reference.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    01 April 2013

  Author:

    Original FORTRAN77 version by Barry Brown, James Lovato.
    C version by John Burkardt.

  Reference:

    Joachim Ahrens, Ulrich Dieter,
    Computer Methods for Sampling From the
    Exponential and Normal Distributions,
    Communications of the ACM,
    Volume 15, Number 10, October 1972, pages 873-882.

  Parameters:

    Output, float SEXPO, a random deviate from the standard
    exponential distribution.
*/
{
  float a;
  int i;
  const float q[8] = {
       0.6931472, 
       0.9333737, 
       0.9888778, 
       0.9984959, 
       0.9998293, 
       0.9999833, 
       0.9999986, 
       0.9999999 };
  float u;
  float umin;
  float ustar;
  float value;

  a = 0.0;
  u = r4_uni_01 ( );

  for ( ; ; )
  {
    u = u + u;

    if ( 1.0 < u )
    {
      break;
    }
    a = a + q[0];
  }

  u = u - 1.0;

  if ( u <= q[0] )
  {
    value = a + u;
    return value;
  }

  i = 0;
  ustar = r4_uni_01 ( );
  umin = ustar;

  for ( ; ; )
  {
    ustar = r4_uni_01 ( );
    umin = r4_min ( umin, ustar );
    i = i + 1;

    if ( u <= q[i] )
    {
      break;
    }
  }

  value = a + umin * q[0];

  return value;
}
/******************************************************************************/

float sgamma ( float a )

/******************************************************************************/
/*
  Purpose:

    SGAMMA samples the standard Gamma distribution.

  Discussion:

    This procedure corresponds to algorithm GD in the reference.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    01 April 2013

  Author:

    Original FORTRAN77 version by Barry Brown, James Lovato.
    C version by John Burkardt.

  Reference:

    Joachim Ahrens, Ulrich Dieter,
    Generating Gamma Variates by a Modified Rejection Technique,
    Communications of the ACM,
    Volume 25, Number 1, January 1982, pages 47-54.

  Parameters:

    Input, float A, the parameter of the standard gamma
    distribution.  0.0 < A < 1.0.

    Output, float SGAMMA, a random deviate from the distribution.
*/
{
  const float a1 =  0.3333333;
  const float a2 = -0.2500030;
  const float a3 =  0.2000062;
  const float a4 = -0.1662921;
  const float a5 =  0.1423657;
  const float a6 = -0.1367177;
  const float a7 =  0.1233795;
  float b;
  float c;
  float d;
  float e;
  const float e1 = 1.0;
  const float e2 = 0.4999897;
  const float e3 = 0.1668290;
  const float e4 = 0.0407753;
  const float e5 = 0.0102930;
  float p;
  float q;
  float q0;
  const float q1 =  0.04166669;
  const float q2 =  0.02083148;
  const float q3 =  0.00801191;
  const float q4 =  0.00144121;
  const float q5 = -0.00007388;
  const float q6 =  0.00024511;
  const float q7 =  0.00024240;
  float r;
  float s;
  float s2;
  float si;
  const float sqrt32 = 5.656854;
  float t;
  float u;
  float v;
  float value;
  float w;
  float x;

  if ( 1.0 <= a )
  {
    s2 = a - 0.5;
    s = sqrt ( s2 );
    d = sqrt32 - 12.0 * s;
/*
  Immediate acceptance.
*/
    t = snorm ( );
    x = s + 0.5 * t;
    value = x * x;

    if ( 0.0 <= t )
    {
      return value;
    }
/*
  Squeeze acceptance.
*/
    u = r4_uni_01 ( );
    if ( d * u <= t * t * t )
    {
      return value;
    }

    r = 1.0 / a;
    q0 = (((((( q7 
      * r + q6 ) 
      * r + q5 ) 
      * r + q4 ) 
      * r + q3 ) 
      * r + q2 ) 
      * r + q1 ) 
      * r;
/*
  Approximation depending on size of parameter A.
*/
    if ( 13.022 < a )
    {
      b = 1.77;
      si = 0.75;
      c = 0.1515 / s;
    }
    else if ( 3.686 < a )
    {
      b = 1.654 + 0.0076 * s2;
      si = 1.68 / s + 0.275;
      c = 0.062 / s + 0.024;
    }
    else
    {
      b = 0.463 + s + 0.178 * s2;
      si = 1.235;
      c = 0.195 / s - 0.079 + 0.16 * s;
    }
/*
  Quotient test.
*/
    if ( 0.0 < x )
    {
      v = 0.5 * t / s;

      if ( 0.25 < fabs ( v ) )
      {
        q = q0 - s * t + 0.25 * t * t + 2.0 * s2 * log ( 1.0 + v );
      }
      else
      {
        q = q0 + 0.5 * t * t * (((((( a7 
          * v + a6 ) 
          * v + a5 ) 
          * v + a4 ) 
          * v + a3 ) 
          * v + a2 ) 
          * v + a1 ) 
          * v;
      }

      if ( log ( 1.0 - u ) <= q )
      {
        return value;
      }
    }

    for ( ; ; )
    {
      e = sexpo ( );
      u = 2.0 * r4_uni_01 ( ) - 1.0;
 
      if ( 0.0 <= u )
      {
        t = b + fabs ( si * e );
      }
      else
      {
        t = b - fabs ( si * e );
      }
/*
  Possible rejection.
*/
      if ( t < -0.7187449 )
      {
        continue;
      }
/*
  Calculate V and quotient Q.
*/
      v = 0.5 * t / s;

      if ( 0.25 < fabs ( v ) )
      {
        q = q0 - s * t + 0.25 * t * t + 2.0 * s2 * log ( 1.0 + v );
      }
      else
      {
        q = q0 + 0.5 * t * t * (((((( a7 
          * v + a6 ) 
          * v + a5 ) 
          * v + a4 ) 
          * v + a3 ) 
          * v + a2 ) 
          * v + a1 ) 
          *  v;
      }
/*
  Hat acceptance.
*/
      if ( q <= 0.0 )
      {
        continue;
      }

      if ( 0.5 < q )
      {
        w = exp ( q ) - 1.0;
      }
      else
      {
        w = (((( e5 * q + e4 ) * q + e3 ) * q + e2 ) * q + e1 ) * q;
      }
/*
  May have to sample again.
*/
      if ( c * fabs ( u ) <= w * exp ( e - 0.5 * t * t ) )
      {
        break;
      }
    }

    x = s + 0.5 * t;
    value = x * x;

    return value;
  }
/*
  Method for A < 1.
*/
  else
  {
    b = 1.0 + 0.3678794 * a;

    for ( ; ; )
    {
      p = b * r4_uni_01 ( );

      if ( p < 1.0 )
      {
        value = exp ( log ( p ) / a );

        if ( value <= sexpo ( ) )
        {
          return value;
        }
        continue;
      }
      value = - log ( ( b - p ) / a );

      if ( ( 1.0 - a ) * log ( value ) <= sexpo ( ) )
      {
        break;
      }
    }
  }
  return value;
}
/******************************************************************************/

float snorm (void)

/******************************************************************************/
/*
  Purpose:

    SNORM samples the standard normal distribution.

  Discussion:

    This procedure corresponds to algorithm FL, with M = 5, in the reference.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    01 April 2013

  Author:

    Original FORTRAN77 version by Barry Brown, James Lovato.
    C version by John Burkardt.

  Reference:

    Joachim Ahrens, Ulrich Dieter,
    Extensions of Forsythe's Method for Random
    Sampling from the Normal Distribution,
    Mathematics of Computation,
    Volume 27, Number 124, October 1973, page 927-937.

  Parameters:

    Output, float SNORM, a random deviate from the distribution.
*/
{
  const float a[32] = {
        0.0000000, 0.3917609E-01, 0.7841241E-01, 0.1177699, 
        0.1573107, 0.1970991,     0.2372021,     0.2776904, 
        0.3186394, 0.3601299,     0.4022501,     0.4450965, 
        0.4887764, 0.5334097,     0.5791322,     0.6260990, 
        0.6744898, 0.7245144,     0.7764218,     0.8305109, 
        0.8871466, 0.9467818,     1.009990,      1.077516, 
        1.150349,  1.229859,      1.318011,      1.417797, 
        1.534121,  1.675940,      1.862732,      2.153875 };
  float aa;
  const float d[31] = {
        0.0000000, 0.0000000, 0.0000000, 0.0000000, 
        0.0000000, 0.2636843, 0.2425085, 0.2255674, 
        0.2116342, 0.1999243, 0.1899108, 0.1812252, 
        0.1736014, 0.1668419, 0.1607967, 0.1553497, 
        0.1504094, 0.1459026, 0.1417700, 0.1379632, 
        0.1344418, 0.1311722, 0.1281260, 0.1252791, 
        0.1226109, 0.1201036, 0.1177417, 0.1155119, 
        0.1134023, 0.1114027, 0.1095039 };
  const float h[31] = {
        0.3920617E-01, 0.3932705E-01, 0.3950999E-01, 0.3975703E-01, 
        0.4007093E-01, 0.4045533E-01, 0.4091481E-01, 0.4145507E-01, 
        0.4208311E-01, 0.4280748E-01, 0.4363863E-01, 0.4458932E-01, 
        0.4567523E-01, 0.4691571E-01, 0.4833487E-01, 0.4996298E-01, 
        0.5183859E-01, 0.5401138E-01, 0.5654656E-01, 0.5953130E-01, 
        0.6308489E-01, 0.6737503E-01, 0.7264544E-01, 0.7926471E-01, 
        0.8781922E-01, 0.9930398E-01, 0.1155599,     0.1404344, 
        0.1836142,     0.2790016,     0.7010474 };
  int i;
  float s;
  const float t[31] = {
        0.7673828E-03, 0.2306870E-02, 0.3860618E-02, 0.5438454E-02, 
        0.7050699E-02, 0.8708396E-02, 0.1042357E-01, 0.1220953E-01, 
        0.1408125E-01, 0.1605579E-01, 0.1815290E-01, 0.2039573E-01, 
        0.2281177E-01, 0.2543407E-01, 0.2830296E-01, 0.3146822E-01, 
        0.3499233E-01, 0.3895483E-01, 0.4345878E-01, 0.4864035E-01, 
        0.5468334E-01, 0.6184222E-01, 0.7047983E-01, 0.8113195E-01, 
        0.9462444E-01, 0.1123001,     0.1364980,     0.1716886, 
        0.2276241,     0.3304980,     0.5847031 };
  float tt;
  float u;
  float ustar;
  float value;
  float w;
  float y;

  u = r4_uni_01 ( );
  if ( u <= 0.5 )
  {
    s = 0.0;
  }
  else
  {
    s = 1.0;
  }
  u = 2.0 * u - s;
  u = 32.0 * u;
  i = ( int ) ( u );
  if ( i == 32 )
  {
    i = 31;
  }
/*
  Center
*/
  if ( i != 0 )
  {
    ustar = u - ( float ) ( i );
    aa = a[i-1];

    for ( ; ; )
    {
      if ( t[i-1] < ustar )
      {
        w = ( ustar - t[i-1] ) * h[i-1];

        y = aa + w;

        if ( s != 1.0 )
        {
          value = y;
        }
        else
        {
          value = -y;
        }
        return value;
      }
      u = r4_uni_01 ( );
      w = u * ( a[i] - aa );
      tt = ( 0.5 * w + aa ) * w;

      for ( ; ; )
      {
        if ( tt < ustar )
        {
          y = aa + w;
          if ( s != 1.0 )
          {
            value = y;
          }
          else
          {
            value = -y;
          }
          return value;
        }

        u = r4_uni_01 ( );

        if ( ustar < u )
        {
          break;
        }
        tt = u;
        ustar = r4_uni_01 ( );
      }
      ustar = r4_uni_01 ( );
    }
  }
/*
  Tail
*/
  else
  {
    i = 6;
    aa = a[31];

    for ( ; ; )
    {
      u = u + u;

      if ( 1.0 <= u )
      {
        break;
      }
      aa = aa + d[i-1];
      i = i + 1;
    }

    u = u - 1.0;
    w = u * d[i-1];
    tt = ( 0.5 * w + aa ) * w;

    for ( ; ; )
    {
      ustar = r4_uni_01 ( );

      if ( tt < ustar )
      {
        y = aa + w;
        if ( s != 1.0 )
        {
          value = y;
        }
        else
        {
          value = -y;
        }
        return value;
      }

      u = r4_uni_01 ( );

      if ( u <= ustar )
      {
        tt = u;
      }
      else
      {
        u = r4_uni_01 ( );
        w = u * d[i-1];
        tt = ( 0.5 * w + aa ) * w;
      }
    }
  }
}
/******************************************************************************/

int spofa ( float a[], int lda, int n )

/******************************************************************************/
/*
  Purpose:

    SPOFA factors a real symmetric positive definite matrix.

  Discussion:

    SPOFA is usually called by SPOCO, but it can be called
    directly with a saving in time if RCOND is not needed.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    04 April 2006

  Author:

    C version by John Burkardt.

  Reference:

    Jack Dongarra, Cleve Moler, Jim Bunch and Pete Stewart,
    LINPACK User's Guide,
    SIAM, (Society for Industrial and Applied Mathematics),
    3600 University City Science Center,
    Philadelphia, PA, 19104-2688.
    ISBN 0-89871-172-X

  Parameters:

    Input/output, float A[LDA*N].  On input, the symmetric matrix
    to be  factored.  Only the diagonal and upper triangle are used.
    On output, an upper triangular matrix R so that A = R'*R
    where R' is the transpose.  The strict lower triangle is unaltered.
    If INFO /= 0, the factorization is not complete.

    Input, int LDA, the leading dimension of the array A.

    Input, int N, the order of the matrix.

    Output, int SPOFA, error flag.
    0, for normal return.
    K, signals an error condition.  The leading minor of order K is not
    positive definite.
*/
{
  int info;
  int j;
  int k;
  float s;
  float t;

  for ( j = 1; j <= n; j++ )
  {
    s = 0.0;

    for ( k = 1; k <= j-1; k++ )
    {
      t = a[k-1+(j-1)*lda] - sdot ( k-1, a+0+(k-1)*lda, 1, a+0+(j-1)*lda, 1 );
      t = t / a[k-1+(k-1)*lda];
      a[k-1+(j-1)*lda] = t;
      s = s + t * t;
    }

    s = a[j-1+(j-1)*lda] - s;

    if ( s <= 0.0 )
    {
      info = j;
      return info;
    }

    a[j-1+(j-1)*lda] = sqrt ( s );
  }

  info = 0;

  return info;
}
/******************************************************************************/

void stats ( float x[], int n, float *av, float *var, float *xmin, float *xmax )

/******************************************************************************/
/*
  Purpose:

    STATS computes statistics for a given array.

  Discussion:

    This procedure computes the average and variance of an array.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    01 April 2013

  Author:

    Original FORTRAN77 version by Barry Brown, James Lovato.
    C version by John Burkardt.

  Parameters:

    Input, float X[N], the array to be analyzed.

    Input, int N, the dimension of the array.

    Output, float *AV, the average value.

    Output, float *VAR, the variance.

    Output, float *XMIN, *XMAX, the minimum and maximum entries.
*/
{
  int i;
  float total;

  *xmin = x[0];
  *xmax = x[0];
  total = x[0];
  for ( i = 1; i < n; i++ )
  {
    total = total + x[i];
    *xmin = r4_min ( *xmin, x[i] );
    *xmax = r4_max ( *xmax, x[i] );
  }

  *av = total / ( float ) ( n );

  total = 0.0;
  for ( i = 0; i < n; i++ )
  {
    total = total + pow ( x[i] - *av, 2 );
  }
  *var = total / ( float ) ( n - 1 );

  return;
}
/******************************************************************************/

void trstat ( char *pdf, float parin[], float *av, float *var )

/******************************************************************************/
/*
  Purpose:

    TRSTAT returns the mean and variance for distributions.

  Discussion:

    Although I really want to do a case-insensitive comparison, for some
    reason STRCMPI is NOT available to C, while STRCMP is.

    This procedure returns the mean and variance for a number of statistical
    distributions as a function of their parameters.

    The input vector PARIN is used to pass in the parameters necessary
    to specify the distribution.  The number of these parameters varies
    per distribution, and it is necessary to specify an ordering for the
    parameters used to a given distribution.  The ordering chosen here
    is as follows:

    bet
      PARIN(1) is A
      PARIN(2) is B
    bin
      PARIN(1) is Number of trials
      PARIN(2) is Prob Event at Each Trial
    chi
      PARIN(1) = df
    exp
      PARIN(1) = mu
    f
      PARIN(1) is df numerator
      PARIN(2) is df denominator
    gam
      PARIN(1) is A
      PARIN(2) is R
    nbn
      PARIN(1) is N
      PARIN(2) is P
    nch
      PARIN(1) is df
      PARIN(2) is noncentrality parameter
    nf
      PARIN(1) is df numerator
      PARIN(2) is df denominator
      PARIN(3) is noncentrality parameter
    nor
      PARIN(1) is mean
      PARIN(2) is standard deviation
    poi
      PARIN(1) is Mean
    unf
      PARIN(1) is LOW bound
      PARIN(2) is HIGH bound

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    02 April 2013

  Author:

    Original FORTRAN77 version by Barry Brown, James Lovato.
    C version by John Burkardt.

  Parameters:

    Input, char *PDF, indicates the distribution:
    'bet'  beta distribution
    'bin'  binomial
    'chi'  chisquare
    'exp'  exponential
    'f'    F (variance ratio)
    'gam'  gamma
    'nbn'  negative binomial
    'nch'  noncentral chisquare
    'nf'   noncentral f
    'nor'  normal
    'poi'  Poisson
    'unf'  uniform

    Input, float PARIN[*], the parameters of the distribution.

    Output, float *AV, the mean of the specified distribution.

    Output, float *VAR, the variance of the specified distribuion.
*/
{
  float a;
  float b;
  int n;
  float p;
  float r;
  float width;

  if ( s_eqi ( pdf, "bet" ) )
  {
    *av = parin[0] / ( parin[0] + parin[1] );
    *var = ( *av * parin[1] ) / ( ( parin[0] + parin[1] ) * 
      ( parin[0] + parin[1] + 1.0 ) );
  }
  else if ( s_eqi ( pdf, "bin" ) )
  {
    n = ( int ) ( parin[0] );
    p = parin[1];
    *av = ( float ) ( n ) * p;
    *var = ( float ) ( n ) * p * ( 1.0 - p );
  }
  else if ( s_eqi ( pdf, "chi" ) )
  {
    *av = parin[0];
    *var = 2.0 * parin[0];
  }
  else if ( s_eqi ( pdf, "exp" ) )
  {
    *av = parin[0];
    *var = pow ( parin[0], 2 );
  }
  else if ( s_eqi ( pdf, "f" ) )
  {
    if ( parin[1] <= 2.0001 )
    {
      *av = -1.0;
    }
    else
    {
      *av = parin[1] / ( parin[1] - 2.0 );
    }

    if ( parin[1] <= 4.0001 )
    {
      *var = -1.0;
    }
    else
    {
      *var = ( 2.0 * pow ( parin[1], 2 ) * ( parin[0] + parin[1] - 2.0 ) ) / 
        ( parin[0] * pow ( parin[1] - 2.0, 2 ) * ( parin[1] - 4.0 ) );
    }
  }
  else if ( s_eqi ( pdf, "gam" ) )
  {
    a = parin[0];
    r = parin[1];
    *av = r / a;
    *var = r / a / a;
  }
  else if ( s_eqi ( pdf, "nbn" ) )
  {
    n = ( int ) ( parin[0] );
    p = parin[1];
    *av = n * ( 1.0 - p ) / p;
    *var = n * ( 1.0 - p ) / p / p;
  }
  else if ( s_eqi ( pdf, "nch" ) )
  {
    a = parin[0] + parin[1];
    b = parin[1] / a;
    *av = a;
    *var = 2.0 * a * ( 1.0 + b );
  }
  else if ( s_eqi ( pdf, "nf" ) )
  {
    if ( parin[1] <= 2.0001 )
    {
      *av = -1.0;
    }
    else
    {
      *av = ( parin[1] * ( parin[0] + parin[2] ) ) 
        / ( ( parin[1] - 2.0 ) * parin[0] );
    }

    if ( parin[1] <= 4.0001 )
    {
      *var = -1.0;
    }
    else
    {
      a = pow ( parin[0] + parin[2], 2 )
        + ( parin[0] + 2.0 * parin[2] ) * ( parin[1] - 2.0 );
      b = pow ( parin[1] - 2.0, 2 ) * ( parin[1] - 4.0 );
      *var = 2.0 * pow ( parin[1] / parin[0], 2 ) * ( a / b );
    }
  }
  else if ( s_eqi ( pdf, "nor" ) )
  {
    *av = parin[0];
    *var = pow ( parin[1], 2 );
  }
  else if ( s_eqi ( pdf, "poi" ) )
  {
    *av = parin[0];
    *var = parin[0];
  }
  else if ( s_eqi ( pdf, "unf" ) )
  {
    width = parin[1] - parin[0];
    *av = parin[0] + width / 2.0;
    *var = width * width / 12.0;
  }
  else
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "TRSTAT - Fatal error!\n" );
    fprintf ( stderr, "  Illegal input value for PDF.\n" );
    exit ( 1 );
  }
  return;
}

