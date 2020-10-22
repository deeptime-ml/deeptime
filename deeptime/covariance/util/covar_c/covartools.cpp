/********************************************************************************
 * This file is part of scikit-time.                                            *
 *                                                                              *
 * Copyright (c) 2020 AI4Science Group, Freie Universitaet Berlin (GER)         *
 *                                                                              *
 * scikit-time is free software: you can redistribute it and/or modify          *
 * it under the terms of the GNU Lesser General Public License as published by  *
 * the Free Software Foundation, either version 3 of the License, or            *
 * (at your option) any later version.                                          *
 *                                                                              *
 * This program is distributed in the hope that it will be useful,              *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of               *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                *
 * GNU General Public License for more details.                                 *
 *                                                                              *
 * You should have received a copy of the GNU Lesser General Public License     *
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.        *
 ********************************************************************************/

#include <pybind11/pybind11.h>

#include "covartools.hpp"

namespace py = pybind11;


PYBIND11_MODULE(_covartools, m) {
    m.doc() = "covariance computation utilities.";

    // ================================================
    // Check for constant columns
    // ================================================
    m.def("variable_cols_char", &_variable_cols<char>);
    m.def("variable_cols_int", &_variable_cols<int>);
    m.def("variable_cols_long", &_variable_cols<long>);
    m.def("variable_cols_float", &_variable_cols<float>);
    m.def("variable_cols_double", &_variable_cols<double>);
}
