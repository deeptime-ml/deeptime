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

#ifndef __util_h_
#define __util_h_

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>


int my_isinf(double x) {
#if _MSC_VER && !__INTEL_COMPILER
	return ! _finite(x);
#else
	return isinf(x);
#endif
}

int my_isnan(double x) {
#if _MSC_VER && !__INTEL_COMPILER
	return _isnan(x);
#else
	return isnan(x);
#endif
}

/**
    Helper function, tests if x is numerically positive

    :param x:
    :return:
*/
int
is_positive(double x)
{
    double eps = 1e-15;
    if (x >= eps && !my_isinf(x) && !my_isnan(x))
		return 1;
    else
    	return 0;
}

double my_fmin(double a, double b) {
#if _MSC_VER && !__INTEL_COMPILER
	return __min(a,b);
#else
	return fmin(a,b);
#endif
}


#endif
