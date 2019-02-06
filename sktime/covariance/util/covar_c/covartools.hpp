#pragma once

#include <cstdlib>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;


/** Checks each column whether it is constant in the rows or not

@param cols : (N) result array that will be filled with 0 (column constant) or 1 (column variable)
@param X : (M, N) array
@param M : int
@param N : int

*/
template<typename dtype>
int _variable_cols(py::array_t<bool, py::array::c_style> &np_cols,
                   const py::array_t<dtype, py::array::c_style> &np_X,
                   float tol=0, std::size_t min_constant=0) {
    // compare first and last row to get constant candidates
    std::size_t i, j;
    std::size_t ro;
    std::size_t M = np_X.shape(0), N = np_X.shape(1);
    dtype diff;
    std::size_t nconstant = N;  // current number of constant columns
    auto cols = np_cols.mutable_data(0);
    auto X = np_X.data(0);
    // by default all 0 (constant)
    for (j = 0; j < N; j++)
        cols[j] = false;

    // go through all rows in order to confirm constant candidates
    for (i = 0; i < M; i++) {
        ro = i * N;
        for (j = 0; j < N; j++) {
            if (! cols[j]) {
                // note: the compiler will eliminate this branch, if dtype != (float, double)
                if (std::is_floating_point<dtype>::value) {
                    diff = std::abs(X[j] - X[ro + j]);
                    if (diff >= tol) {
                        cols[j] = true;
                        nconstant--;
                        // are constant columns below threshold? Then interrupt.
                        if (nconstant < min_constant)
                            return 0;
                        // do we have 0 constant columns? Then we can stop regularly.
                        if (nconstant == 0)
                            return 1;
                    }
                } else {
                    if (X[j] != X[ro + j]) {
                        cols[j] = true;
                        nconstant--;
                        if (nconstant < min_constant)
                            return 0;
                        if (nconstant == 0)
                            return 1;
                    }
                }
            }
        }
    }
    return 1;
}
