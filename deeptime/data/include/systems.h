#pragma once

#include <vector>
#include <random>
#include <chrono>

#include "common.h"
#include "integrator.h"

namespace py = pybind11;

struct ode_tag {
};
struct sde_tag {
};

template<class T, std::size_t DIM>
using Vector = std::array<T, DIM>;

template<typename T, std::size_t DIM>
using Matrix = Vector<Vector<T, DIM>, DIM>;

//------------------------------------------------------------------------------
// ABC flow
//------------------------------------------------------------------------------
template<typename T>
struct ABCFlow {
    using system_type = ode_tag;

    static constexpr std::size_t DIM = 3;
    using dtype = T;
    using State = Vector<T, DIM>;
    using Integrator = deeptime::RungeKutta<State, DIM>;

    constexpr State f(const State &x) const {
        return {{
                        a_ * std::sin(x[2]) + c_ * std::cos(x[1]),
                        b_ * std::sin(x[0]) + a_ * std::cos(x[2]),
                        c_ * std::sin(x[1]) + b_ * std::cos(x[0])
                }};
    }

    double h{1e-3};
    std::size_t nSteps{1000};
private:
    static constexpr T a_ = 1.73205080757; // sqrt(3)
    static constexpr T b_ = 1.41421356237;  // sqrt(2.);
    static constexpr T c_ = 1;
};

//------------------------------------------------------------------------------
// Ornstein-Uhlenbeck process
//------------------------------------------------------------------------------
template<typename T>
struct OrnsteinUhlenbeck {
    using system_type = sde_tag;

    static constexpr std::size_t DIM = 1;
    using dtype = T;
    using State = Vector<dtype, DIM>;
    using Integrator = deeptime::EulerMaruyama<State, DIM>;

    constexpr dtype energy(const State &x) const {
        return 0.5 * alpha * x[0] * x[0];
    }

    constexpr State f(const State &x) const {
        return {{-alpha * x[0]}};
    }

    static constexpr T alpha = 1.;
    static constexpr T beta = 4.;
    static constexpr Matrix<T, 1> sigma{{{{2 / beta}}}};

    T h{1e-3};
    std::size_t nSteps{500};
};

//------------------------------------------------------------------------------
// Simple triple-well in one dimension, use interval [0, 6]
//------------------------------------------------------------------------------
template<typename T>
struct TripleWell1D {
    using system_type = sde_tag;

    static constexpr std::size_t DIM = 1;
    using dtype = T;
    using State = Vector<T, DIM>;
    using Integrator = deeptime::EulerMaruyama<State, DIM>;

    constexpr dtype energy(const State &x) const {
        return -(24.82*x[0] - 41.4251*x[0]*x[0] + 27.5344*std::pow(x[0], 3)
               - 8.53128*std::pow(x[0], 4) + 1.24006 * std::pow(x[0], 5) - 0.0684 * std::pow(x[0], 6)) + 5;
    }

    constexpr State f(const State &x) const {
        return {{
                        -1 * (-24.82002100 + 82.85029600 * x[0] - 82.6031550 * x[0] * x[0]
                              + 34.125104 * std::pow(x[0], 3) - 6.20030 * std::pow(x[0], 4) +
                              0.4104 * std::pow(x[0], 5))
                }};
    }

    static constexpr Matrix<T, 1> sigma{{{{0.75}}}};
    T h{1e-3};
    std::size_t nSteps{500};
};


//------------------------------------------------------------------------------
// Double well problem
//------------------------------------------------------------------------------
template<typename T>
struct DoubleWell2D {
    using system_type = sde_tag;

    static constexpr std::size_t DIM = 2;
    using dtype = T;
    using State = Vector<T, DIM>;
    using Integrator = deeptime::EulerMaruyama<State, DIM>;

    constexpr dtype energy(const State &x) const {
        return (x[0]*x[0]-1.) * (x[0]*x[0]-1.) + x[1] * x[1];
    }

    constexpr State f(const State &x) const {
        return {{-4 * x[0] * x[0] * x[0] + 4 * x[0], -2 * x[1]}};
    }

    static constexpr Matrix<T, 2> sigma{{{{0.7, 0.0}}, {{0.0, 0.7}}}};
    T h{1e-3};
    std::size_t nSteps{10000};
};

//------------------------------------------------------------------------------
// Quadruple well problem
//------------------------------------------------------------------------------
template<typename T>
struct QuadrupleWell2D {
    using system_type = sde_tag;

    static constexpr std::size_t DIM = 2;
    using dtype = T;
    using State = Vector<T, DIM>;
    using Integrator = deeptime::EulerMaruyama<State, DIM>;

    constexpr dtype energy(const State &x) const {
        return (x[0]*x[0] - 1)*(x[0]*x[0] - 1) + (x[1]*x[1] - 1)*(x[1]*x[1] - 1);
    }

    constexpr State f(const State &x) const {
        // Quadruple well potential: V = (x(1, :).^2 - 1).^2 + (x(2, :).^2 - 1).^2
        return {{-4 * x[0] * x[0] * x[0] + 4 * x[0], -4 * x[1] * x[1] * x[1] + 4 * x[1]}};
    }

    static constexpr T s = 0.70710678118; // sqrt(.5);
    static constexpr Matrix<T, 2> sigma{{{{s, 0.0}}, {{0.0, s}}}};
    T h{1e-3};
    std::size_t nSteps{10000};
};


//------------------------------------------------------------------------------
// Unsymmetric quadruple well problem
//------------------------------------------------------------------------------
template<typename T>
struct QuadrupleWellAsymmetric2D {
    using system_type = sde_tag;

    static constexpr std::size_t DIM = 2;
    using dtype = T;
    using State = Vector<T, DIM>;
    using Integrator = deeptime::EulerMaruyama<State, DIM>;

    constexpr dtype energy(const State &x) const {
        return + x[0]*x[0]*x[0]*x[0] - (1. / 16.) * x[0]*x[0]*x[0] - 2.*x[0]*x[0] + (3./16.) * x[0]
               + x[1]*x[1]*x[1]*x[1] - (1. / 8.) * x[1]*x[1]*x[1] - 2*x[1]*x[1] + (3./8.) * x[1];
    }

    constexpr State f(const State &x) const {
        return {{
                        -4 * x[0] * x[0] * x[0] + (3.0 / 16.0) * x[0] * x[0] + 4 * x[0] - 3.0 / 16.0,
                        -4 * x[1] * x[1] * x[1] + (3.0 / 8.0) * x[1] * x[1] + 4 * x[1] - 3.0 / 8.0
                }};
    }

    static constexpr Matrix<T, 2> sigma{{{{0.6, 0.0}}, {{0.0, 0.6}}}};
    T h{1e-3};
    std::size_t nSteps{10000};
};

//------------------------------------------------------------------------------
// Triple well problem
//------------------------------------------------------------------------------
template<typename T>
struct TripleWell2D {
    using system_type = sde_tag;

    static constexpr std::size_t DIM = 2;
    using dtype = T;
    using State = Vector<T, DIM>;
    using Integrator = deeptime::EulerMaruyama<State, DIM>;
    
    constexpr dtype energy(const State &x) const {
        const auto& xv = x[0];
        const auto& yv = x[1];
        return + 3.*std::exp(- (xv * xv) - (yv - 1./3.)*(yv - 1./3.))
               - 3.*std::exp(- (xv * xv) - (yv - 5./3.)*(yv - 5./3.))
               - 5.*std::exp(-(xv - 1.)*(xv - 1.) - yv * yv)
               - 5.*std::exp(-(xv + 1.)*(xv + 1.) - yv * yv)
               + (2./10.)*xv*xv*xv*xv
               + (2./10.)*std::pow(yv - 1./3., 4.);
    }

    constexpr State f(const State &x) const {
        return {{
                        -(3 * std::exp(-x[0] * x[0] - (x[1] - 1.0 / 3) * (x[1] - 1.0 / 3)) * (-2 * x[0])
                          - 3 * std::exp(-x[0] * x[0] - (x[1] - 5.0 / 3) * (x[1] - 5.0 / 3)) * (-2 * x[0])
                          - 5 * std::exp(-(x[0] - 1.0) * (x[0] - 1.0) - x[1] * x[1]) * (-2 * (x[0] - 1.0))
                          - 5 * std::exp(-(x[0] + 1.0) * (x[0] + 1.0) - x[1] * x[1]) * (-2 * (x[0] + 1.0))
                          + 8.0 / 10 * std::pow(x[0], 3)),
                        -(3 * std::exp(-x[0] * x[0] - (x[1] - 1.0 / 3) * (x[1] - 1.0 / 3)) * (-2 * (x[1] - 1.0 / 3))
                          - 3 * std::exp(-x[0] * x[0] - (x[1] - 5.0 / 3) * (x[1] - 5.0 / 3)) * (-2 * (x[1] - 5.0 / 3))
                          - 5 * std::exp(-(x[0] - 1.0) * (x[0] - 1.0) - x[1] * x[1]) * (-2 * x[1])
                          - 5 * std::exp(-(x[0] + 1.0) * (x[0] + 1.0) - x[1] * x[1]) * (-2 * x[1])
                          + 8.0 / 10 * std::pow(x[1] - 1.0 / 3, 3))
                }};
    }

    static constexpr Matrix<T, 2> sigma{{{{1.09, 0.0}}, {{0.0, 1.09}}}};
    T h{1e-5};
    std::size_t nSteps{10000};
};


namespace detail {
template<typename T>
std::is_member_pointer<decltype(&T::sigma)> is_member_sigma(int);

template<typename T>
decltype(T::sigma, std::true_type{}) is_member_sigma(long);

template<typename T>
std::false_type is_member_sigma(...);

template<typename T>
using IsMemberSigma = decltype(is_member_sigma<T>(0));

template<typename System>
typename System::State evaluate(const System &system, typename System::Integrator &integrator,
                                const typename System::State &x, double h, std::size_t nSteps, ode_tag) {
    auto rhs = [&](const auto &_x) {
        return system.f(_x);
    };
    return integrator.eval(rhs, h, nSteps, x);
}

template<typename System>
typename System::State evaluate(const System &system, typename System::Integrator &integrator,
                                const typename System::State &x, double h, std::size_t nSteps, sde_tag) {
    auto rhs = [&system](const auto &_x) {
        return system.f(_x);
    };
    if constexpr(detail::IsMemberSigma<System>{}) {
        return integrator.eval(rhs, system.sigma, h, nSteps, x);
    } else {
        return integrator.eval(rhs, System::sigma, h, nSteps, x);
    }
}
}


template<typename System>
typename System::State evaluate(const System &system, typename System::Integrator &integrator,
                                const typename System::State &x, double h, std::size_t nSteps) {
    return detail::evaluate(system, integrator, x, h, nSteps, typename System::system_type());
}

template<typename System>
typename System::State evaluate(const System &system, typename System::Integrator &integrator,
                                const typename System::State &x) {
    return detail::evaluate(system, integrator, x, system.h, system.nSteps, typename System::system_type());
}

template<typename System>
typename System::Integrator createIntegrator(std::int64_t /*seed*/, ode_tag) {
    return typename System::Integrator{};
}

template<typename System>
typename System::Integrator createIntegrator(std::int64_t seed, sde_tag) {
    return typename System::Integrator{seed};
}

template<typename dtype, typename System>
np_array_nfc<dtype> evaluateSystem(const System &system, const np_array_nfc<dtype> &x,
                                   std::int64_t seed = -1, int nThreads = -1) {
    if (seed >= 0 && nThreads != 1) {
        throw std::invalid_argument("Fixing the seed requires setting the number of threads to 1.");
    }
    np_array_nfc<dtype> y({x.shape(0), x.shape(1)});

    auto xBuf = x.template unchecked<2>();
    {
        const auto d = x.shape(1); // dimension of the state space
        if (d != static_cast<decltype(d)>(System::DIM)) {
            std::stringstream errmsg;
            errmsg << "The dynamical system is " << System::DIM << "-dimensional, but the provided test points had "
                   << d << " dimensions.";
            throw std::invalid_argument(errmsg.str());
        }
    }
    const auto nTestPoints = x.shape(0); // number of snapshots
    auto yBuf = y.template mutable_unchecked<2>();

    #if defined(USE_OPENMP)
    if (nThreads > 0) {
        omp_set_num_threads(nThreads);
    }
    #else
    nThreads = 0;
    #endif

    typename System::State testPoint = {};

    // for all test points
    #pragma omp parallel default(none) firstprivate(system, nTestPoints, xBuf, yBuf, testPoint, seed)
    {
        auto integrator = createIntegrator<System>(seed, typename System::system_type());

        #pragma omp for
        for (py::ssize_t i = 0; i < nTestPoints; ++i) {

            // copy new test point into x vector
            for (std::size_t k = 0; k < System::DIM; ++k) {
                testPoint[k] = xBuf(i, k);
            }

            auto yi = evaluate(system, integrator, testPoint); // evaluate dynamical system

            for (std::size_t k = 0; k < System::DIM; ++k) {
                // copy result into y vector
                yBuf(i, k) = yi[k];
            }
        }
    }

    return y;
}

template<typename dtype, typename System>
np_array_nfc<dtype>
trajectory(System &system, const np_array_nfc<dtype> &x, std::size_t length, std::int64_t seed = -1) {
    np_array_nfc<dtype> y({length, System::DIM});
    {
        const auto d = x.shape(1);
        if (d != System::DIM) {
            std::stringstream errmsg;
            errmsg << "The dynamical system is " << System::DIM << "-dimensional, but the provided test points had "
                   << d << " dimensions.";
            throw std::invalid_argument(errmsg.str());
        }
        const auto m = x.shape(0);
        if (m != 1) {
            throw std::invalid_argument("Currently only supports one test point.");
        }
    }

    auto xBuf = x.template unchecked<2>();
    auto yBuf = y.template mutable_unchecked<2>();

    auto integrator = createIntegrator<System>(seed, typename System::system_type());

    for (size_t k = 0; k < System::DIM; ++k) {
        // copy initial condition
        yBuf(0, k) = xBuf(0, k);
    }

    typename System::State testPoint;
    for (size_t i = 1; i < length; ++i) {
        for (size_t k = 0; k < System::DIM; ++k) {
            // copy new test point into x vector
            testPoint[k] = yBuf(i - 1, k);
        }

        // evaluate dynamical system
        auto yi = evaluate(system, integrator, testPoint);

        // copy result into y vector
        for (size_t k = 0; k < System::DIM; ++k) {
            yBuf(i, k) = yi[k];
        }
    }
    return y;
}
