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

/* * moduleauthor:: F. Noe <frank DOT noe AT fu-berlin DOT de>
 */

#ifndef _sample_rev_h_
#define _sample_rev_h_

double _update_step(double v0, double v1, double v2, double c0, double c1, double c2, int random_walk_stepsize);
double _sum_all(double* X, int n);
void _normalize_all(double* X, int n);
void _normalize_all_sparse(double* X, int* I, int* J, int n, int n_idx);
double _sum_row(double* X, int n, int i);
void _update(double* C, double* sumC, double* X, int n, int n_step);
void _update_sparse_sparse(double* C, double* sumC, double* X, double* sumX, int* I, int* J, int n, int n_idx, int n_step);

void _print_matrix(double* X, int n);
void _update_sparse_speedtest(double* C, double* sumC, double* X, double* sumX, int* I, int* J, int n, int n_idx, int n_step);


// declarations
void _update_sparse(double* C, double* sumC, double* X, double* sumX, int* I, int* J, int n, int n_idx, int n_step);
void _generate_row_indexes(int* I, int n, int n_idx, int* row_indexes);
void _normalize_all(double* X, int n);


#endif
