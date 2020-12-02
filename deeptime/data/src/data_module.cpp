// author: clonker, sklus

#include <functional>
#include <utility>
#include <type_traits>

#include "common.h"
#include "pbf.h"
#include "systems.h"

namespace detail{
template< class... >
using void_t = void;

template< class, class = void >
struct system_has_potential : std::false_type { };
template< class T >
struct system_has_potential<T, void_t<decltype(std::declval<T>().energy(std::declval<typename T::State>()))>> : std::true_type { };
}

template<typename T>
static constexpr bool system_has_potential_v = detail::system_has_potential<T>::value;

using namespace pybind11::literals;

using dtype = float;
static constexpr int DIM = 2;
using PBF = deeptime::pbf::PBF<DIM, dtype>;

template<typename T, std::size_t dim>
struct PyODE {
    using system_type = ode_tag;

    static constexpr std::size_t DIM = dim;
    using dtype = T;
    using State = Vector<T, DIM>;
    using Integrator = deeptime::RungeKutta<State, DIM>;
    using Rhs = std::function<State(State)>;

    explicit PyODE(Rhs &&rhs) : rhs(std::move(rhs)) {}

    [[nodiscard]] State f(const State &x) const {
        py::gil_scoped_acquire gil;
        return rhs(x);
    }

    Rhs rhs {};
    T h {1e-3};
    std::size_t nSteps {1000};
};

template<typename T, std::size_t dim>
struct PySDE {
    using system_type = sde_tag;

    static constexpr std::size_t DIM = dim;
    using dtype = T;
    using State = Vector<T, DIM>;
    using Integrator = deeptime::EulerMaruyama<State, DIM, T>;
    using Rhs = std::function<State(State)>;
    using Sigma = Matrix<T, DIM>;

    PySDE(const Sigma &sigma, Rhs &&rhs) : sigma(sigma), rhs(std::move(rhs)) {}

    [[nodiscard]] State f(const State &x) const {
        return rhs(x);
    }

    Sigma sigma;
    Rhs rhs {};

    T h {1e-3};
    std::size_t nSteps {1000};
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
template<typename System, typename... InitArgs>
auto exportSystem(py::module& m, const std::string &name) {
    using npDtype = typename System::dtype;
    auto clazz = py::class_<System>(m, name.c_str())
            .def(py::init<InitArgs...>())
            .def_readwrite("h", &System::h)
            .def_readwrite("n_steps", &System::nSteps)
            .def_readonly_static("dimension", &System::DIM)
            .def("__call__", [](System &self, const np_array_nfc<npDtype> &x, std::int64_t seed, int nThreads) -> np_array_nfc<npDtype> {
                return evaluateSystem(self, x, seed, nThreads);
            }, "test_points"_a, "seed"_a = -1, "n_jobs"_a = 1)
            .def("trajectory", [](System &self, const np_array_nfc<npDtype> &x, std::size_t length, std::int64_t seed) -> np_array_nfc<npDtype> {
                return trajectory(self, x, length, seed);
            }, "x0"_a, "n_evaluations"_a, "seed"_a = -1);
    if constexpr(system_has_potential_v<System>) {
        clazz.def("potential", [](System &self, const np_array_nfc<npDtype> &x) {
            auto nPoints = static_cast<std::size_t>(x.shape(0));
            np_array_nfc<npDtype> y (nPoints);

            auto yBuf = y.template mutable_unchecked<1>();
            std::fill(y.mutable_data(), y.mutable_data() + nPoints, 0.);

            auto xBuf = x.template unchecked<2>();

            typename System::State testPoint;
            for (std::size_t i = 0; i < nPoints; ++i) {
                for (std::size_t k = 0; k < System::DIM; ++k) {
                    testPoint[k] = xBuf(i, k);
                }
                yBuf(i) = self.energy(testPoint);
            }

            return y;
        });
    }

    return clazz;
}

template<std::size_t DIM>
void exportPyODE(py::module& m, const std::string& name) {
    using PyODE = PyODE<double, DIM>;
    exportSystem<PyODE, typename PyODE::Rhs>(m, name);
}

template<std::size_t DIM>
void exportPySDE(py::module& m, const std::string& name) {
    using PySDE = PySDE<double, DIM>;
    exportSystem<PySDE, typename PySDE::Sigma, typename PySDE::Rhs>(m, name);
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
    exportSystem<ABCFlow<double>>(m, "ABCFlow");
    exportSystem<OrnsteinUhlenbeck<double>>(m, "OrnsteinUhlenbeck");
    exportSystem<TripleWell1D<double>>(m, "TripleWell1D");
    exportSystem<DoubleWell2D<double>>(m, "DoubleWell2D");
    exportSystem<QuadrupleWell2D<double>>(m, "QuadrupleWell2D");
    exportSystem<TripleWell2D<double>>(m, "TripleWell2D");
    exportSystem<QuadrupleWellAsymmetric2D<double>>(m, "QuadrupleWellAsymmetric2D");

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
