// author: clonker

#include <pybind11/pybind11.h>

#include "common.h"
#include "pbf.h"

using dtype = float;
static constexpr int DIM = 2;
using PBF = deeptime::pbf::PBF<DIM, dtype>;

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
    py::class_<PBF>(m, "PBF").def(py::init(&makePbf))
        .def("predict_positions", &PBF::predictPositions)
        .def("update_neighborlist", &PBF::updateNeighborlist)
        .def("calculate_lambdas", &PBF::calculateLambdas)
        .def("run", [](PBF& self, std::uint32_t steps, dtype drift) {
            auto traj = self.run(steps, drift);
            np_array<dtype> npTraj ({static_cast<std::size_t>(steps), static_cast<std::size_t>(DIM*self.nParticles())});
            std::copy(traj.begin(), traj.end(), npTraj.mutable_data());
            return npTraj;
        })
        .def_property("n_solver_iterations", &PBF::nSolverIterations, &PBF::setNSolverIterations)
        .def_property("gravity", &PBF::gravity, &PBF::setGravity)
        .def_property("timestep", &PBF::dt, &PBF::setDt)
        .def_property("epsilon", &PBF::epsilon, &PBF::setEpsilon)
        .def_property("rest_density", &PBF::rho0, &PBF::setRho0)
        .def_property("tensile_instability_distance", &PBF::tensileInstabilityDistance,
                      &PBF::setTensileInstabilityDistance)
        .def_property("tensile_instability_k", &PBF::tensileInstabilityK, &PBF::setTensileInstabilityK)
        .def_property_readonly("n_particles", &PBF::nParticles)
        .def_property_readonly("domain_size", &PBF::gridSize);
}
