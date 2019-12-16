/**
 * @file bhmm_output_models_module.cpp
 * @brief 
 * @authors noe, clonker
 * @date 12/16/19
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

template<typename dtype>
using np_array = py::array_t<dtype, py::array::c_style>;

namespace gaussian {

template<typename dtype>
void pObs(const np_array<dtype> &obs, const np_array<dtype> &mus, const np_array<dtype> &sigmas) {
    auto N = static_cast<std::size_t>(mus.shape(0));
    auto T = static_cast<std::size_t>(obs.shape(0));

}

}

PYBIND11_MODULE(_bhmm_output_models, m) {
    {
        auto discrete = m.def_submodule("discrete");
        // todo
    }

    {
        auto gaussian = m.def_submodule("gaussian");
        // todo
    }
}
