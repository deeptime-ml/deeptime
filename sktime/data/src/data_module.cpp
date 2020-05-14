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

PBF makePbf(np_array<dtype> pos, np_array<dtype> gridSize, dtype interactionRadius, int nJobs) {
    if(pos.ndim() != 2) {
        throw std::invalid_argument("position array must be 2-dimensional");
    }
    std::array<dtype, 2> gridSizeArr {{ gridSize.at(0), gridSize.at(1) }};
    std::vector<dtype> particles;
    auto* ptr = pos.mutable_data();
    auto nParticles = pos.shape(0);
    particles.reserve(static_cast<std::size_t>(nParticles * DIM));
    std::copy(ptr, ptr + nParticles * DIM, std::back_inserter(particles));

    if(pos.shape(1) != DIM) {
        throw std::invalid_argument("shape(1) of position array must match dim=" + std::to_string(DIM));
    }
    return {particles, static_cast<std::size_t>(nParticles), gridSizeArr, interactionRadius, nJobs};
}

PYBIND11_MODULE(_data_bindings, m) {
    m.def("voodoo", [](np_array<dtype> pos, np_array<dtype> gridSize, dtype interactionRadius, int nJobs) {
        {
            auto pbf = makePbf(pos, gridSize, interactionRadius, nJobs);
        }
    });
    py::class_<PBF, std::unique_ptr<PBF>>(m, "PBF").def(py::init(&makePbf), py::keep_alive<1, 2>())
        .def("predict_positions", &PBF::predictPositions)
        .def("update_neighborlist", &PBF::updateNeighborlist)
        .def("calculate_lambdas", &PBF::calculateLambdas)
        .def("run", &PBF::run)
        .def_property("n_solver_iterations", &PBF::nSolverIterations, &PBF::setNSolverIterations)
        .def_property("gravity", &PBF::gravity, &PBF::setGravity)
        .def_property("timestep", &PBF::dt, &PBF::setDt)
        .def_property("epsilon", &PBF::epsilon, &PBF::setEpsilon)
        .def_property("equilibrium_density", &PBF::rho0, &PBF::setRho0)
        .def_property("tensile_instability_scale", &PBF::tensileInstabilityScale, &PBF::setTensileInstabilityScale)
        .def_property("tensile_instability_k", &PBF::tensileInstabilityK, &PBF::setTensileInstabilityK);
}
