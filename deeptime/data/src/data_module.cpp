// author: clonker, sklus

#include <functional>
#include <utility>

#include "common.h"
#include "pbf.h"
#include "systems.h"

using dtype = float;
static constexpr int DIM = 2;
using PBF = deeptime::pbf::PBF<DIM, dtype>;

template<typename T, typename State>
class PyODE : public ODE<::PyODE, T, State> {
public:
    explicit PyODE(std::function<State(State)> rhs, T h = 1e-3, std::size_t nSteps = 1000)
            : rhs(std::move(rhs)), _h(h), _nSteps(nSteps) {}

    State f(const State &x) {
        return rhs(x);
    }

    [[nodiscard]] T h() const { return _h; }

    [[nodiscard]] std::size_t nSteps() const { return _nSteps; }

private:
    std::function<State(State)> rhs;

    T _h;
    std::size_t _nSteps;
};

template<typename State>
struct DimFromState {
    constexpr static std::size_t value = std::tuple_size<State>::value;
};

template<typename T, typename State>
class PySDE : public SDE<::PySDE, T, State>, public EulerMaruyamaIntegrator<State, DimFromState<State>::value, T> {
public:
    explicit PySDE(std::function<State(State)> rhs, Matrix<T, DimFromState<State>::value> sigma,
                   double h, std::size_t nSteps, std::int64_t seed)
                   : EulerMaruyamaIntegrator<State, DimFromState<State>::value, T>(seed), _h(h), _nSteps(nSteps),
                     sigma(sigma), rhs(std::move(rhs)) {
    }

    State f(const State &x) {
        return rhs(x);
    }

    [[nodiscard]] T h() const { return _h; }

    [[nodiscard]] std::size_t nSteps() const { return _nSteps; }

    double _h;
    std::size_t _nSteps;
    Matrix<T, DimFromState<State>::value> sigma;
    std::function<State(State)> rhs;
};


PBF makePbf(np_array<dtype> pos, const np_array<dtype>& gridSize, dtype interactionRadius, int nJobs) {
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

template<typename System, typename ... InitArgs>
void exportSystem(py::module& m, const std::string &name) {
    py::class_<System>(m, name.c_str())
            .def(py::init<InitArgs...>())
            .def_property_readonly("dimension", &System::dim)
            .def("__call__", &System::operator())
            .def("trajectory", &System::trajectory);
}

template<std::size_t DIM>
void exportPyODE(py::module& m, const std::string& name) {
    using State = Vector<double, DIM>;
    using PyODE = PyODE<double, State>;
    using Rhs = std::function<State(State)>;
    exportSystem<PyODE, Rhs, double, std::size_t>(m, name);
}

template<std::size_t DIM>
void exportPySDE(py::module& m, const std::string& name) {
    using State = Vector<double, DIM>;
    using Sigma = Matrix<double, DIM>;
    using PySDE = PySDE<double, State>;
    using Rhs = std::function<State(State)>;
    exportSystem<PySDE, Rhs, Sigma, double, std::size_t, std::int64_t>(m, name);
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

    // more examples can be found at: https://github.com/sklus/d3s/tree/master/cpp
    exportSystem<ABCFlow<double>, double, std::size_t>(m, "ABCFlow");
    exportSystem<OrnsteinUhlenbeck<double>, std::int64_t, double, std::size_t>(m, "OrnsteinUhlenbeck");
    exportSystem<TripleWell1D<double>, std::int64_t, double, std::size_t>(m, "TripleWell1D");
    exportSystem<DoubleWell2D<double>, std::int64_t, double, std::size_t>(m, "DoubleWell2D");
    exportSystem<QuadrupleWell2D<double>, std::int64_t, double, std::size_t>(m, "QuadrupleWell2D");
    exportSystem<TripleWell2D<double>, std::int64_t, double, std::size_t>(m, "TripleWell2D");
    exportSystem<QuadrupleWellUnsymmetric2D<double>, std::int64_t, double, std::size_t>(m, "QuadrupleWellUnsymmetric2D");

    exportPyODE<1>(m, "PyODE1D");
    exportPyODE<2>(m, "PyODE2D");
    exportPyODE<3>(m, "PyODE3D");
    exportPyODE<4>(m, "PyODE4D");
    exportPyODE<5>(m, "PyODE5D");

    exportPySDE<1>(m, "PySDE1D");
    exportPySDE<2>(m, "PySDE2D");
    exportPySDE<3>(m, "PySDE3D");
    exportPySDE<4>(m, "PySDE4D");
    exportPySDE<5>(m, "PySDE5D");
}
