/**
 *
 *
 * @file bindings.cpp
 * @brief 
 * @author clonker
 * @date 2/1/19
 */

#include <pybind11/pybind11.h>
#include <sktime-data/sktime-data.h>

PYBIND11_MODULE(sktime_data_bindings, m) {
    m.def("moo", []() -> std::string { return "moo"; });
}
