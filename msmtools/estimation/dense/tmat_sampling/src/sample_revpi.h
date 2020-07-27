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
#ifndef _TMATRIX_SAMPLING_REVPI_
#define _TMATRIX_SAMPLING_REVPI_

extern double sample_quad(double xkl, double xkk, double xll,
			  double ckl, double clk, double ckk, double cll,
			  double bk, double bl);

extern double sample_quad_rw(double xkl, double xkk, double xll,
			  double ckl, double clk, double ckk, double cll,
			  double bk, double bl);

extern void update(double *X, double *C, double *b, size_t n);

extern void update_sparse(double *X, double *C, double *b, size_t n,
			  size_t * I, size_t * J, size_t n_idx);

#endif /* _TMATRIX_SAMPLING_REVPI_ */
