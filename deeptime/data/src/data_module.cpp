// author: clonker, sklus

#include <pybind11/pybind11.h>

#include "common.h"
#include "pbf.h"
#include "systems.h"

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

template<typename SystemType>
void exportSystem(py::module& m, std::string&& name) {
    py::class_<SystemType>(m, name.c_str())
            .def(py::init<typename SystemType::dtype, std::size_t>())
            .def_property_readonly("dimension", &SystemType::getDimension())
            .def("__call__", &SystemType::operator())
            .def("simulate", &SystemType::getTrajectory());
}

#define S(x) #x
#define EXPORT_DISC(name)                               \
    py::class_<name>(m, pname)                          \
        .def(py::init<>())                              \
        .def("getDimension", &name::getDimension)       \
        .def("__call__", &name::operator())             \
        .def("getTrajectory", &name::getTrajectory);
#define EXPORT_CONT(name, pname)                        \
    py::class_<name>(m, pname)                          \
        .def(py::init<double, size_t>())                \
        .def("getDimension", &name::getDimension)       \
        .def("__call__", &name::operator())             \
        .def("getTrajectory", &name::getTrajectory);


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

    // more examples can be found at: https://github.com/sklus/d3s/tree/master/cpp
    EXPORT_CONT(ABCFlow<double>, "ABCFlow");
    EXPORT_CONT(OrnsteinUhlenbeck<double>, "OrnsteinUhlenbeck");
    EXPORT_CONT(TripleWell1D<double>, "TripleWell1D");
    EXPORT_CONT(DoubleWell2D<double>, "DoubleWell2D");
    EXPORT_CONT(QuadrupleWell2D<double>, "QuadrupleWell2D");
    EXPORT_CONT(QuadrupleWellUnsymmetric2D<double>, "QuadrupleWellUnsymmetric2D");
    EXPORT_CONT(TripleWell2D<double>, "TripleWell2D");
}
