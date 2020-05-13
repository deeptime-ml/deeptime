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

PBF makePbf(np_array<dtype> pos, np_array<dtype> domain, np_array<int> gridSize, int nJobs) {
    if(pos.ndim() != 2) {
        throw std::invalid_argument("position array must be 2-dimensional");
    }
    std::array<dtype, 2> domainArr {{ domain.at(0), domain.at(1) }};
    std::array<std::uint32_t, 2> gridSizeArr {{static_cast<std::uint32_t>(gridSize.at(0)),
                                               static_cast<std::uint32_t>(gridSize.at(1)) }};
    auto* ptr = pos.mutable_data();
    auto nParticles = pos.shape(0);
    if(pos.shape(1) != DIM) {
        throw std::invalid_argument("shape(1) of position array must match dim=" + std::to_string(DIM));
    }
    return {ptr, static_cast<std::size_t>(nParticles), domainArr, gridSizeArr, nJobs};
}

PYBIND11_MODULE(_data_bindings, m) {
    py::class_<PBF>(m, "PBF").def(py::init(&makePbf))
        .def("predict_positions", &PBF::predictPositions)
        .def("update_neighborlist", &PBF::updateNeighborlist)
        .def("calculate_lambdas", &PBF::calculateLambdas)
        .def_property("n_solver_iterations", &PBF::nSolverIterations, &PBF::setNSolverIterations)
        .def_property("gravity", &PBF::gravity, &PBF::setGravity)
        .def_property("timestep", &PBF::dt, &PBF::setDt)
        .def_property("epsilon", &PBF::epsilon, &PBF::setEpsilon)
        .def_property("equilibrium_density", &PBF::rho0, &PBF::setRho0);
}
