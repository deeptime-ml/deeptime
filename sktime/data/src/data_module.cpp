/**
 *
 *
 * @file data_bindings.cpp
 * @brief 
 * @author clonker
 * @date 5/13/20
 */

#include <pybind11/pybind11.h>

#include "common.h"
#include "pbf.h"

using dtype = float;
static constexpr int DIM = 2;
using PBF = sktime::pbf::PBF<DIM, dtype>;

PBF makePbf(np_array<dtype> pos, np_array<dtype> domain, dtype gravity, dtype dt, int nJobs) {
    if(pos.ndim() != 2) {
        throw std::invalid_argument("position array must be 2-dimensional");
    }
    std::array<dtype, 2> domainArr {{ domain.at(0), domain.at(1) }};
    auto* ptr = pos.mutable_data();
    auto nParticles = pos.shape(0);
    if(pos.shape(1) != DIM) {
        throw std::invalid_argument("shape(1) of position array must match dim=" + std::to_string(DIM));
    }
    return {ptr, static_cast<std::size_t>(nParticles), domainArr, gravity, dt, nJobs};
}

PYBIND11_MODULE(_data_bindings, m) {
    py::class_<PBF>(m, "PBF").def(py::init(&makePbf))
        .def("predict_positions", &PBF::predictPositions);
}
