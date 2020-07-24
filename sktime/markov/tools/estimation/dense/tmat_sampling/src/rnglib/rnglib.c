# include <stdlib.h>
# include <stdio.h>
# include <time.h>

# include "rnglib.h"

/******************************************************************************/

void advance_state ( int k )

/******************************************************************************/
/*
  Purpose:

    ADVANCE_STATE advances the state of the current generator.

  Discussion:

    This procedure advances the state of the current generator by 2^K 
    values and resets the initial seed to that value.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    30 March 2013

  Author:

    Original Pascal version by Pierre L'Ecuyer, Serge Cote.
    C version by John Burkardt.

  Reference:

    Pierre LEcuyer, Serge Cote,
    Implementing a Random Number Package with Splitting Facilities,
    ACM Transactions on Mathematical Software,
    Volume 17, Number 1, March 1991, pages 98-111.

  Parameters:

    Input, int K, indicates that the generator is to be 
    advanced by 2^K values.
    0 <= K.
*/
{
  const int a1 = 40014;
  const int a2 = 40692;
  int b1;
  int b2;
  int cg1;
  int cg2;
  int g;
  int i;
  const int m1 = 2147483563;
  const int m2 = 2147483399;

  if ( k < 0 )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "ADVANCE_STATE - Fatal error!\n" );
    fprintf ( stderr, "  Input exponent K is out of bounds.\n" );
    exit ( 1 );
  }
/*
  Check whether the package must be initialized.
*/
  if ( ! initialized_get ( ) )
  {
    printf ( "\n" );
    printf ( "ADVANCE_STATE - Note:\n" );
    printf ( "  Initializing RNGLIB package.\n" );
    initialize ( );
  }
/*
  Get the current generator index.
*/
  g = cgn_get ( );

  b1 = a1;
  b2 = a2;

  for ( i = 1; i <= k; k++ )
  {
    b1 = multmod ( b1, b1, m1 );
    b2 = multmod ( b2, b2, m2 );
  }

  cg_get ( g, &cg1, &cg2 );
  cg1 = multmod ( b1, cg1, m1 );
  cg2 = multmod ( b2, cg2, m2 );
  cg_set ( g, cg1, cg2 );

  return;
}
/******************************************************************************/

int antithetic_get (void )

/******************************************************************************/
/*
  Purpose:

    ANTITHETIC_GET queries the antithetic value for a given generator.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    01 April 2013

  Author:

    John Burkardt

  Parameters:

    Output, int ANTITHETIC_GET, is TRUE (1) if generator G is antithetic.
*/
{
  int i;
  int value;

  i = -1;
  antithetic_memory ( i, &value );

  return value;
}
/******************************************************************************/

void antithetic_memory ( int i, int *value )

/******************************************************************************/
/*
  Purpose:

    ANTITHETIC_MEMORY stores the antithetic value for all generators.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    01 April 2013

  Author:

    John Burkardt

  Parameters:

    Input, int I, the desired action.
    -1, get a value.
    0, initialize all values.
    1, set a value.

    Input/output, int *VALUE.  For I = -1, VALUE is an output
    quantity.  If I = +1, then VALUE is an input quantity.
*/
{
# define G_MAX 32

  static int a_save[G_MAX];
  int g;
  const int g_max = 32;
  int j;

  if ( i < 0 )
  {
    g = cgn_get ( );
    *value = a_save[g];
  }
  else if ( i == 0 )
  {
    for ( j = 0; j < g_max; j++ )
    {
      a_save[j] = 0;
    }
  }
  else if ( 0 < i )
  {
    g = cgn_get ( );
    a_save[g] = *value;
  }

  return;
# undef G_MAX
}
/******************************************************************************/

void antithetic_set ( int value )

/******************************************************************************/
/*
  Purpose:

    ANTITHETIC_SET sets the antithetic value for a given generator.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    01 April 2013

  Author:

    John Burkardt

  Parameters:

    Input, int VALUE, is TRUE (1) if generator G is to be antithetic.
*/
{
  int i;

  i = +1;
  antithetic_memory ( i, &value );

  return;
}
/******************************************************************************/

void cg_get ( int g, int *cg1, int *cg2 )

/******************************************************************************/
/*
  Purpose:

    CG_GET queries the CG values for a given generator.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    27 March 2013

  Author:

    John Burkardt

  Parameters:

    Input, int G, the index of the generator.
    0 <= G <= 31.

    Output, int *CG1, *CG2, the CG values for generator G.
*/
{
  int i;

  i = -1;
  cg_memory ( i, g, cg1, cg2 );

  return;
}
/******************************************************************************/

void cg_memory ( int i, int g, int *cg1, int *cg2 )

/******************************************************************************/
/*
  Purpose:

    CG_MEMORY stores the CG values for all generators.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    30 March 2013

  Author:

    John Burkardt

  Parameters:

    Input, int I, the desired action.
    -1, get a value.
    0, initialize all values.
    1, set a value.

    Input, int G, for I = -1 or +1, the index of 
    the generator, with 0 <= G <= 31.

    Input/output, int *CG1, *CG2.  For I = -1, 
    these are output, for I = +1, these are input, for I = 0,
    these arguments are ignored.  When used, the arguments are
    old or new values of the CG parameter for generator G.
*/
{
# define G_MAX 32

  static int cg1_save[G_MAX];
  static int cg2_save[G_MAX];
  const int g_max = 32;
  int j;

  if ( g < 0 || g_max <= g )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "CG_MEMORY - Fatal error!\n" );
    fprintf ( stderr, "  Input generator index G is out of bounds.\n" );
    exit ( 1 );
  }

  if ( i < 0 )
  {
    *cg1 = cg1_save[g];
    *cg2 = cg2_save[g];
  }
  else if ( i == 0 )
  {
    for ( j = 0; j < g_max; j++ )
    {
      cg1_save[j] = 0;
      cg2_save[j] = 0;
    }
  }
  else if ( 0 < i )
  {
    cg1_save[g] = *cg1;
    cg2_save[g] = *cg2;
  }

  return;
# undef G_MAX
}
/******************************************************************************/

void cg_set ( int g, int cg1, int cg2 )

/******************************************************************************/
/*
  Purpose:

    CG_SET sets the CG values for a given generator.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    27 March 2013

  Author:

    John Burkardt

  Parameters:

    Input, int G, the index of the generator.
    0 <= G <= 31.

    Input, int CG1, CG2, the CG values for generator G.
*/
{
  int i;

  i = +1;
  cg_memory ( i, g, &cg1, &cg2 );

  return;
}
/******************************************************************************/

int cgn_get (void)

/******************************************************************************/
/*
  Purpose:

    CGN_GET gets the current generator index

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    30 March 2013

  Author:

    John Burkardt

  Parameters:

    Output, int CGN_GET, the current generator index.
*/
{
  int g;
  int i;

  i = -1;
  cgn_memory ( i, &g );

  return g;
}
/******************************************************************************/

void cgn_memory ( int i, int *g )

/******************************************************************************/
/*
  Purpose:

    CGN_MEMORY stores the current generator index.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    30 March 2013

  Author:

    John Burkardt

  Parameters:

    Input, int I, the desired action.
    -1, get the value.
    0, initialize the value.
    1, set the value.

    Input/output, int *G.  For I = -1 or 0, this is output.
    For I = 1, this is input.
*/
{
# define G_MAX 32

  static int g_save = 0;
  const int g_max = 32;
  int j;

  if ( i < 0 )
  {
    *g = g_save;
  }
  else if ( i == 0 )
  {
    g_save = 0;
    *g = g_save;
  }
  else if ( 0 < i )
  {

    if ( *g < 0 || g_max <= *g )
    {
      fprintf ( stderr, "\n" );
      fprintf ( stderr, "CGN_MEMORY - Fatal error!\n" );
      fprintf ( stderr, "  Input generator index G is out of bounds.\n" );
      exit ( 1 );
    }

    g_save = *g;
  }

  return;
# undef G_MAX
}
/******************************************************************************/

void cgn_set ( int g )

/******************************************************************************/
/*
  Purpose:

    CGN_SET sets the current generator index.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    30 March 2013

  Author:

    John Burkardt

  Parameters:

    Input, int G, the current generator index.
    0 <= G <= 31.
*/
{
  int i;

  i = +1;
  cgn_memory ( i, &g );

  return;
}
/******************************************************************************/

void get_state ( int *cg1, int *cg2 )

/******************************************************************************/
/*
  Purpose:

    GET_STATE returns the state of the current generator.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    30 March 2013

  Author:

    Original Pascal version by Pierre L'Ecuyer, Serge Cote.
    C version by John Burkardt.

  Reference:

    Pierre LEcuyer, Serge Cote,
    Implementing a Random Number Package with Splitting Facilities,
    ACM Transactions on Mathematical Software,
    Volume 17, Number 1, March 1991, pages 98-111.

  Parameters:

    Output, int *CG1, *CG2, the CG values for the current generator.
*/
{
  int g;
/*
  Check whether the package must be initialized.
*/
  if ( ! initialized_get ( ) )
  {
    printf ( "\n" );
    printf ( "GET_STATE - Note:\n" );
    printf ( "  Initializing RNGLIB package.\n" );
    initialize ( );
  }
/*
  Get the current generator index.
*/
  g = cgn_get ( );
/*
  Retrieve the seed values for this generator.
*/
  cg_get ( g, cg1, cg2 );

  return;
}
/******************************************************************************/

int i4_uni (void)

/******************************************************************************/
/*
  Purpose:

    I4_UNI generates a random positive integer.

  Discussion:

    This procedure returns a random integer following a uniform distribution 
    over (1, 2147483562) using the current generator.

    The original name of this function was "random()", but this conflicts
    with a standard library function name in C.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    05 August 2013

  Author:

    Original Pascal version by Pierre L'Ecuyer, Serge Cote.
    C version by John Burkardt.

  Reference:

    Pierre LEcuyer, Serge Cote,
    Implementing a Random Number Package with Splitting Facilities,
    ACM Transactions on Mathematical Software,
    Volume 17, Number 1, March 1991, pages 98-111.

  Parameters:

    Output, int I4_UNI, the random integer.
*/
{
  const int a1 = 40014;
  const int a2 = 40692;
  int cg1;
  int cg2;
  int g;
  int k;
  const int m1 = 2147483563;
  const int m2 = 2147483399;
  int value;
  int z;
/*
  Check whether the package must be initialized.
*/
  if ( ! initialized_get ( ) )
  {
    printf ( "\n" );
    printf ( "I4_UNI - Note:\n" );
    printf ( "  Initializing RNGLIB package.\n" );
    initialize ( );
  }
/*
  Get the current generator index.
*/
  g = cgn_get ( );
/*
  Retrieve the current seeds.
*/
  cg_get ( g, &cg1, &cg2 );
/*
  Update the seeds.
*/
  k = cg1 / 53668;
  cg1 = a1 * ( cg1 - k * 53668 ) - k * 12211;

  if ( cg1 < 0 )
  {
    cg1 = cg1 + m1;
  }

  k = cg2 / 52774;
  cg2 = a2 * ( cg2 - k * 52774 ) - k * 3791;

  if ( cg2 < 0 )
  {
    cg2 = cg2 + m2;
  }
/*
  Store the updated seeds.
*/
  cg_set ( g, cg1, cg2 );
/*
  Form the random integer.
*/
  z = cg1 - cg2;

  if ( z < 1 )
  {
    z = z + m1 - 1;
  }
/*
  If the generator is antithetic, reflect the value.
*/
  value = antithetic_get ( );

  if ( value )
  {
    z = m1 - z;
  }
  return z;
}
/******************************************************************************/

void ig_get ( int g, int *ig1, int *ig2 )

/******************************************************************************/
/*
  Purpose:

    IG_GET queries the IG values for a given generator.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    27 March 2013

  Author:

    John Burkardt

  Parameters:

    Input, int G, the index of the generator.
    0 <= G <= 31.

    Output, int *IG1, *IG2, the IG values for generator G.
*/
{
  int i;

  i = -1;
  ig_memory ( i, g, ig1, ig2 );

  return;
}
/******************************************************************************/

void ig_memory ( int i, int g, int *ig1, int *ig2 )

/******************************************************************************/
/*
  Purpose:

    IG_MEMORY stores the IG values for all generators.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    30 March 2013

  Author:

    John Burkardt

  Parameters:

    Input, int I, the desired action.
    -1, get a value.
    0, initialize all values.
    1, set a value.

    Input, int G, for I = -1 or +1, the index of 
    the generator, with 0 <= G <= 31.

    Input/output, int *IG1, *IG2.  For I = -1, 
    these are output, for I = +1, these are input, for I = 0,
    these arguments are ignored.  When used, the arguments are
    old or new values of the IG parameter for generator G.
*/
{
# define G_MAX 32

  const int g_max = 32;
  static int ig1_save[G_MAX];
  static int ig2_save[G_MAX];
  int j;

  if ( g < 0 || g_max <= g )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "IG_MEMORY - Fatal error!\n" );
    fprintf ( stderr, "  Input generator index G is out of bounds.\n" );
    exit ( 1 );
  }

  if ( i < 0 )
  {
    *ig1 = ig1_save[g];
    *ig2 = ig2_save[g];
  }
  else if ( i == 0 )
  {
    for ( j = 0; j < g_max; j++ )
    {
      ig1_save[j] = 0;
      ig2_save[j] = 0;
    }
  }
  else if ( 0 < i )
  {
    ig1_save[g] = *ig1;
    ig2_save[g] = *ig2;
  }

  return;
# undef G_MAX
}
/******************************************************************************/

void ig_set ( int g, int ig1, int ig2 )

/******************************************************************************/
/*
  Purpose:

    IG_SET sets the IG values for a given generator.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    27 March 2013

  Author:

    John Burkardt

  Parameters:

    Input, int G, the index of the generator.
    0 <= G <= 31.

    Input, int IG1, IG2, the IG values for generator G.
*/
{
  int i;

  i = +1;
  ig_memory ( i, g, &ig1, &ig2 );

  return;
}
/******************************************************************************/

void init_generator ( int t )

/******************************************************************************/
/*
  Purpose:

    INIT_GENERATOR sets the state of generator G to initial, last or new seed.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    01 April 2013

  Author:

    Original Pascal version by Pierre L'Ecuyer, Serge Cote.
    C version by John Burkardt.

  Reference:

    Pierre LEcuyer, Serge Cote,
    Implementing a Random Number Package with Splitting Facilities,
    ACM Transactions on Mathematical Software,
    Volume 17, Number 1, March 1991, pages 98-111.

  Parameters:

    Input, int T, the seed type:
    0, use the seed chosen at initialization time.
    1, use the last seed.
    2, use a new seed set 2^30 values away.
*/
{
  const int a1_w = 1033780774;
  const int a2_w = 1494757890;
  int cg1;
  int cg2;
  int g;
  int ig1;
  int ig2;
  int lg1;
  int lg2;
  const int m1 = 2147483563;
  const int m2 = 2147483399;
/*
  Check whether the package must be initialized.
*/
  if ( ! initialized_get ( ) )
  {
    printf ( "\n" );
    printf ( "INIT_GENERATOR - Note:\n" );
    printf ( "  Initializing RNGLIB package.\n" );
    initialize ( );
  }
/*
  Get the current generator index.
*/
  g = cgn_get ( );
/*
  0: restore the initial seed.
*/
  if ( t == 0 )
  {
    ig_get ( g, &ig1, &ig2 );
    lg1 = ig1;
    lg2 = ig2;
    lg_set ( g, lg1, lg2 );
  }
/*
  1: restore the last seed.
*/
  else if ( t == 1 )
  {
    lg_get ( g, &lg1, &lg2 );
  }
/*
  2: advance to a new seed.
*/
  else if ( t == 2 )
  {
    lg_get ( g, &lg1, &lg2 );
    lg1 = multmod ( a1_w, lg1, m1 );
    lg2 = multmod ( a2_w, lg2, m2 );
    lg_set ( g, lg1, lg2 );
  }
  else
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "INIT_GENERATOR - Fatal error!\n" );
    fprintf ( stderr, "  Input parameter T out of bounds.\n" );
    exit ( 1 );
  }
/*
  Store the new seed.
*/
  cg1 = lg1;
  cg2 = lg2;
  cg_set ( g, cg1, cg2 );

  return;
}
/******************************************************************************/

void initialize (void)

/******************************************************************************/
/*
  Purpose:

    INITIALIZE initializes the random number generator library.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    01 April 2013

  Author:

    Original Pascal version by Pierre L'Ecuyer, Serge Cote.
    C version by John Burkardt.

  Reference:

    Pierre LEcuyer, Serge Cote,
    Implementing a Random Number Package with Splitting Facilities,
    ACM Transactions on Mathematical Software,
    Volume 17, Number 1, March 1991, pages 98-111.

  Parameters:

    None
*/
{
  int g;
  const int g_max = 32;
  int ig1;
  int ig2;
  int value;
/*
  Remember that we have called INITIALIZE().
*/
  initialized_set ( );
/*
  Initialize all generators to have FALSE antithetic value.
*/
  value = 0;
  for ( g = 0; g < g_max; g++ )
  {
    cgn_set ( g );
    antithetic_set ( value );
  }
/*
  Set the initial seeds.
*/
  ig1 = 1234567890;
  ig2 = 123456789;
  set_initial_seed ( ig1, ig2 );
/*
  Initialize the current generator index to 0.
*/
  g = 0;
  cgn_set ( g );

  // printf ( "\n" );
  // printf ( "INITIALIZE - Note:\n" );
  // printf ( "  The RNGLIB package has been initialized.\n" );

  return;
}
/******************************************************************************/

int initialized_get (void)

/******************************************************************************/
/*
  Purpose:

    INITIALIZED_GET queries the INITIALIZED value.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    28 March 2013

  Author:

    John Burkardt

  Parameters:

    Output, int INITIALIZED_GET, is TRUE (1) if the package has been initialized.
*/
{
  int i;
  int value;

  i = -1;
  initialized_memory ( i, &value );

  return value;
}
/******************************************************************************/

void initialized_memory ( int i, int *initialized )

/******************************************************************************/
/*
  Purpose:

    INITIALIZED_MEMORY stores the INITIALIZED value for the package.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    28 March 2013

  Author:

    John Burkardt

  Parameters:

    Input, int I, the desired action.
    -1, get the value.
    0, initialize the value.
    1, set the value.

    Input/output, int *INITIALIZED.  For I = -1, this is an output
    quantity.  If I = +1, this is an input quantity.  If I = 0, 
    this is ignored.
*/
{
  static int initialized_save = 0;

  if ( i < 0 )
  {
    *initialized = initialized_save;
  }
  else if ( i == 0 )
  {
    initialized_save = 0;
  }
  else if ( 0 < i )
  {
    initialized_save = *initialized;
  }

  return;
}
/******************************************************************************/

void initialized_set (void )

/******************************************************************************/
/*
  Purpose:

    INITIALIZED_SET sets the INITIALIZED value true.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    28 March 2013

  Author:

    John Burkardt

  Parameters:

    None
*/
{
  int i;
  int initialized;

  i = +1;
  initialized = 1;
  initialized_memory ( i, &initialized );

  return;
}
/******************************************************************************/

void lg_get ( int g, int *lg1, int *lg2 )

/******************************************************************************/
/*
  Purpose:

    LG_GET queries the LG values for a given generator.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    27 March 2013

  Author:

    John Burkardt

  Parameters:

    Input, int G, the index of the generator.
    0 <= G <= 31.

    Output, int *LG1, *LG2, the LG values for generator G.
*/
{
  int i;

  i = -1;
  lg_memory ( i, g, lg1, lg2 );

  return;
}
/******************************************************************************/

void lg_memory ( int i, int g, int *lg1, int *lg2 )

/******************************************************************************/
/*
  Purpose:

    LG_MEMORY stores the LG values for all generators.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    30 March 2013

  Author:

    John Burkardt

  Parameters:

    Input, int I, the desired action.
    -1, get a value.
    0, initialize all values.
    1, set a value.

    Input, int G, for I = -1 or +1, the index of 
    the generator, with 0 <= G <= 31.

    Input/output, int *LG1, *LG2.  For I = -1, 
    these are output, for I = +1, these are input, for I = 0,
    these arguments are ignored.  When used, the arguments are
    old or new values of the LG parameter for generator G.
*/
{
# define G_MAX 32

  const int g_max = 32;

  int j;
  static int lg1_save[G_MAX];
  static int lg2_save[G_MAX];

  if ( g < 0 || g_max <= g )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "LG_MEMORY - Fatal error!\n" );
    fprintf ( stderr, "  Input generator index G is out of bounds.\n" );
    exit ( 1 );
  }

  if ( i < 0 )
  {
    *lg1 = lg1_save[g];
    *lg2 = lg2_save[g];
  }
  else if ( i == 0 )
  {
    for ( j = 0; j < g_max; j++ )
    {
      lg1_save[j] = 0;
      lg2_save[j] = 0;
    }
  }
  else if ( 0 < i )
  {
    lg1_save[g] = *lg1;
    lg2_save[g] = *lg2;
  }
  return;
# undef G_MAX
}
/******************************************************************************/

void lg_set ( int g, int lg1, int lg2 )

/******************************************************************************/
/*
  Purpose:

    LG_SET sets the LG values for a given generator.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    27 March 2013

  Author:

    John Burkardt

  Parameters:

    Input, int G, the index of the generator.
    0 <= G <= 31.

    Input, int LG1, LG2, the LG values for generator G.
*/
{
  int i;

  i = +1;
  lg_memory ( i, g, &lg1, &lg2 );

  return;
}
/******************************************************************************/

int multmod ( int a, int s, int m )

/******************************************************************************/
/*
  Purpose:

    MULTMOD carries out modular multiplication.

  Discussion:

    This procedure returns 

      ( A * S ) mod M

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    27 March 2013

  Author:

    Original Pascal version by Pierre L'Ecuyer, Serge Cote.
    C version by John Burkardt.

  Reference:

    Pierre LEcuyer, Serge Cote,
    Implementing a Random Number Package with Splitting Facilities,
    ACM Transactions on Mathematical Software,
    Volume 17, Number 1, March 1991, pages 98-111.

  Parameters:

    Input, int A, S, M, the arguments.

    Output, int MULTMOD, the value of the product of A and S, 
    modulo M.
*/
{
  int a0;
  int a1;
  const int h = 32768;
  int k;
  int p;
  int q;
  int qh;
  int rh;

  if ( a <= 0 )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "MULTMOD - Fatal error!\n" );
    fprintf ( stderr, "  A <= 0.\n" );
    exit ( 1 );
  }

  if ( m <= a )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "MULTMOD - Fatal error!\n" );
    fprintf ( stderr, "  M <= A.\n" );
    exit ( 1 );
  }

  if ( s <= 0 )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "MULTMOD - Fatal error!\n" );
    fprintf ( stderr, "  S <= 0.\n" );
    exit ( 1 );
  }

  if ( m <= s )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "MULTMOD - Fatal error!\n" );
    fprintf ( stderr, "  M <= S.\n" );
    exit ( 1 );
  }

  if ( a < h )
  {
    a0 = a;
    p = 0;
  }
  else
  {
    a1 = a / h;
    a0 = a - h * a1;
    qh = m / h;
    rh = m - h * qh;

    if ( h <= a1 )
    {
      a1 = a1 - h;
      k = s / qh;
      p = h * ( s - k * qh ) - k * rh;

      while ( p < 0 )
      {
        p = p + m;
      }
    }
    else
    {
      p = 0;
    }

    if ( a1 != 0 )
    {
      q = m / a1;
      k = s / q;
      p = p - k * ( m - a1 * q );

      if ( 0 < p )
      {
        p = p - m;
      }

      p = p + a1 * ( s - k * q );

      while ( p < 0 )
      {
        p = p + m;
      }
    }

    k = p / qh;
    p = h * ( p - k * qh ) - k * rh;

    while ( p < 0 )
    {
      p = p + m;
    }
  }

  if ( a0 != 0 )
  {
    q = m / a0;
    k = s / q;
    p = p - k * ( m - a0 * q );

    if ( 0 < p )
    {
      p = p - m;
    }

    p = p + a0 * ( s - k * q );

    while ( p < 0 )
    {
      p = p + m;
    }
  }
  return p;
}
/******************************************************************************/

float r4_uni_01 (void)

/******************************************************************************/
/*
  Purpose:

    R4_UNI_01 returns a uniform random real number in [0,1].

  Discussion:

    This procedure returns a random floating point number from a uniform 
    distribution over (0,1), not including the endpoint values, using the
    current random number generator.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    05 August 2013

  Author:

    Original Pascal version by Pierre L'Ecuyer, Serge Cote.
    C version by John Burkardt.

  Reference:

    Pierre LEcuyer, Serge Cote,
    Implementing a Random Number Package with Splitting Facilities,
    ACM Transactions on Mathematical Software,
    Volume 17, Number 1, March 1991, pages 98-111.

  Parameters:

    Output, float R4_UNI_01, a uniform random value in [0,1].
*/
{
  int i;
  float value;
/*
  Check whether the package must be initialized.
*/
  if ( ! initialized_get ( ) )
  {
    printf ( "\n" );
    printf ( "R4_UNI_01 - Note:\n" );
    printf ( "  Initializing RNGLIB package.\n" );
    initialize ( );
  }
/*
  Get a random integer.
*/
  i = i4_uni ( );
/*
  Scale it to [0,1].
*/
  value = ( float ) ( i ) * 4.656613057E-10;

  return value;
}
/******************************************************************************/

double r8_uni_01 (void)

/******************************************************************************/
/*
  Purpose:

    R8_UNI_01 returns a uniform random double in [0,1].

  Discussion:

    This procedure returns a random floating point number from a uniform 
    distribution over (0,1), not including the endpoint values, using the
    current random number generator.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    05 August 2013

  Author:

    Original Pascal version by Pierre L'Ecuyer, Serge Cote.
    C version by John Burkardt.

  Reference:

    Pierre LEcuyer, Serge Cote,
    Implementing a Random Number Package with Splitting Facilities,
    ACM Transactions on Mathematical Software,
    Volume 17, Number 1, March 1991, pages 98-111.

  Parameters:

    Output, double R8_UNI_01, a uniform random value in [0,1].
*/
{
  int i;
  double value;
/*
  Check whether the package must be initialized.
*/
  if ( ! initialized_get ( ) )
  {
    printf ( "\n" );
    printf ( "R8_UNI_01 - Note:\n" );
    printf ( "  Initializing RNGLIB package.\n" );
    initialize ( );
  }
/*
  Get a random integer.
*/
  i = i4_uni ( );
/*
  Scale it to [0,1].
*/
  value = ( double ) ( i ) * 4.656613057E-10;

  return value;
}
/******************************************************************************/

void set_initial_seed ( int ig1, int ig2 )

/******************************************************************************/
/*
  Purpose:

    SET_INITIAL_SEED resets the initial seed and state for all generators.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    28 March 2013

  Author:

    Original Pascal version by Pierre L'Ecuyer, Serge Cote.
    C version by John Burkardt.

  Reference:

    Pierre LEcuyer, Serge Cote,
    Implementing a Random Number Package with Splitting Facilities,
    ACM Transactions on Mathematical Software,
    Volume 17, Number 1, March 1991, pages 98-111.

  Parameters:

    Input, int IG1, IG2, the initial seed values 
    for the first generator.
    1 <= IG1 < 2147483563
    1 <= IG2 < 2147483399
*/
{
  const int a1_vw = 2082007225;
  const int a2_vw = 784306273;
  int g;
  const int g_max = 32;
  int i;
  const int m1 = 2147483563;
  const int m2 = 2147483399;
  int t;

  if ( ig1 < 1 || m1 <= ig1 )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "SET_INITIAL_SEED - Fatal error!\n" );
    fprintf ( stderr, "  Input parameter IG1 out of bounds.\n" );
    exit ( 1 );
  }

  if ( ig2 < 1 || m2 <= ig2 )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "SET_INITIAL_SEED - Fatal error!\n" );
    fprintf ( stderr, "  Input parameter IG2 out of bounds.\n" );
    exit ( 1 );
  }
/*
  Because INITIALIZE calls SET_INITIAL_SEED, it's not easy to correct
  the error that arises if SET_INITIAL_SEED is called before INITIALIZE.
  So don't bother trying.
*/
  if ( ! initialized_get ( ) )
  {
    printf ( "\n" );
    printf ( "SET_INITIAL_SEED - Fatal error!\n" );
    printf ( "  The RNGLIB package has not been initialized.\n" );
    exit ( 1 );
  }
/*
  Set the initial seed, then initialize the first generator.
*/
  g = 0;
  cgn_set ( g );

  ig_set ( g, ig1, ig2 );

  t = 0;
  init_generator ( t );
/*
  Now do similar operations for the other generators.
*/
  for ( g = 1; g < g_max; g++ )
  {
    cgn_set ( g );
    ig1 = multmod ( a1_vw, ig1, m1 );
    ig2 = multmod ( a2_vw, ig2, m2 );
    ig_set ( g, ig1, ig2 );
    init_generator ( t );
  }
/*
  Now choose the first generator.
*/
  g = 0;
  cgn_set ( g );

  return;
}
/******************************************************************************/

void set_seed ( int cg1, int cg2 )

/******************************************************************************/
/*
  Purpose:

    SET_SEED resets the initial seed and the state of generator G.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    01 April 2013

  Author:

    Original Pascal version by Pierre L'Ecuyer, Serge Cote.
    C version by John Burkardt.

  Reference:

    Pierre LEcuyer, Serge Cote,
    Implementing a Random Number Package with Splitting Facilities,
    ACM Transactions on Mathematical Software,
    Volume 17, Number 1, March 1991, pages 98-111.

  Parameters:

    Input, int CG1, CG2, the CG values for generator G.
    1 <= CG1 < 2147483563
    1 <= CG2 < 2147483399
*/
{
  int g;
  int i;
  const int m1 = 2147483563;
  const int m2 = 2147483399;
  int t;

  if ( cg1 < 1 || m1 <= cg1 )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "SET_SEED - Fatal error!\n" );
    fprintf ( stderr, "  Input parameter CG1 out of bounds.\n" );
    exit ( 1 );
  }

  if ( cg2 < 1 || m2 <= cg2 )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "SET_SEED - Fatal error!\n" );
    fprintf ( stderr, "  Input parameter CG2 out of bounds.\n" );
    exit ( 1 );
  }
/*
  Check whether the package must be initialized.
*/
  if ( ! initialized_get ( ) )
  {
    printf ( "\n" );
    printf ( "SET_SEED - Note:\n" );
    printf ( "  Initializing RNGLIB package.\n" );
    initialize ( );
  }
/*
  Retrieve the current generator index.
*/
  g = cgn_get ( );
/*
  Set the seeds.
*/
  cg_set ( g, cg1, cg2 );
/*
  Initialize the generator.
*/
  t = 0;
  init_generator ( t );

  return;
}
/******************************************************************************/

void timestamp (void)

/******************************************************************************/
/*
  Purpose:

    TIMESTAMP prints the current YMDHMS date as a time stamp.

  Example:

    31 May 2001 09:45:54 AM

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    24 September 2003

  Author:

    John Burkardt

  Parameters:

    None
*/
{
# define TIME_SIZE 40

  static char time_buffer[TIME_SIZE];
  const struct tm *tm;
  time_t now;

  now = time ( NULL );
  tm = localtime ( &now );

  strftime ( time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm );

  printf ( "%s\n", time_buffer );

  return;
# undef TIME_SIZE
}

