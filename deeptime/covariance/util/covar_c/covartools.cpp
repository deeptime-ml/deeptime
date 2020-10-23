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
