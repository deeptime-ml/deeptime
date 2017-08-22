#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "covartools_cpp.h"

namespace py = pybind11;


PYBIND11_PLUGIN(_covartools) {
    pybind11::module m("_covartools", "covariance computation utilities.");

    // ================================================
    // Check for constant columns
    // ================================================
    m.def("variable_cols_char", &_variable_cols<char>);
    m.def("variable_cols_int", &_variable_cols<int>);
    m.def("variable_cols_long", &_variable_cols<long>);

    m.def("variable_cols_float", &_variable_cols<float>);
    m.def("variable_cols_double", &_variable_cols<double>);
    m.def("variable_cols_approx_float", &_variable_cols_approx<float>);
    m.def("variable_cols_approx_double", &_variable_cols_approx<double>);

    // ================================================
    // Row subtraction
    // ================================================
    m.def("subtract_row_float", &_subtract_row<float>);
    m.def("subtract_row_double", &_subtract_row<double>);
    m.def("subtract_row_float_copy", &_subtract_row_copy<float>);
    m.def("subtract_row_double_copy", &_subtract_row_copy<double>);
    return m.ptr();
}
