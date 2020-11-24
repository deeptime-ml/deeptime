#pragma once

#include <vector>
#include <random>
#include <chrono>

#include "common.h"
#include "integrator.h"

namespace py = pybind11;

template<class T, std::size_t DIM>
using Vector = std::array<T, DIM>;

template<typename T, std::size_t DIM>
using Matrix = Vector<Vector<T, DIM>, DIM>;

template<template<typename, typename> class Derived, typename State, typename dtype>
class ODE;

template<template<typename, typename> class Derived, typename State, typename dtype>
class SDE;

template<typename State, std::size_t DIM, typename T>
struct EulerMaruyamaIntegrator {
    explicit EulerMaruyamaIntegrator(std::int64_t seed) : integrator(seed) {}

    deeptime::EulerMaruyama<State, DIM, T> integrator;
};


//------------------------------------------------------------------------------
// Virtual base class for all dynamical systems
//------------------------------------------------------------------------------
template<template<typename, typename> class Derived, typename dtype, typename State>
class DynamicalSystemInterface {
    constexpr static std::size_t DIM = std::tuple_size<State>::value;
public:

    np_array<dtype> operator()(const np_array<dtype> &x) {
        np_array<dtype> y({x.shape(0), x.shape(1)});

        auto xBuf = x.template unchecked<2>();
        {
            const auto d = x.shape(1); // dimension of the state space
            if (d != DIM) {
                std::stringstream errmsg;
                errmsg << "The dynamical system is " << DIM << "-dimensional, but the provided test points had "
                       << d << " dimensions.";
                throw std::invalid_argument(errmsg.str());
            }
        }
        const auto nTestPoints = x.shape(0); // number of snapshots

        auto yBuf = y.template mutable_unchecked<2>();

        // for all test points
        State testPoint;
        for (py::ssize_t i = 0; i < nTestPoints; ++i) {

            // copy new test point into x vector
            for (std::size_t k = 0; k < DIM; ++k) {
                testPoint[k] = xBuf(i, k);
            }

            auto yi = eval(testPoint); // evaluate dynamical system

            for (std::size_t k = 0; k < DIM; ++k) {
                // copy result into y vector
                yBuf(i, k) = yi[k];
            }
        }

        return y;
    }

    np_array<dtype> trajectory(const np_array<dtype> &x, std::size_t length) {
        np_array<dtype> y({length, DIM});
        {
            const auto d = x.shape(1);
            if (d != DIM) {
                std::stringstream errmsg;
                errmsg << "The dynamical system is " << DIM << "-dimensional, but the provided test points had "
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

        for (size_t k = 0; k < DIM; ++k) {
            // copy initial condition
            yBuf(0, k) = xBuf(0, k);
        }

        State testPoint;
        for (size_t i = 1; i < length; ++i) {
            for (size_t k = 0; k < DIM; ++k) {
                // copy new test point into x vector
                testPoint[k] = yBuf(i-1, k);
            }

            // evaluate dynamical system
            auto yi = eval(testPoint);

            // copy result into y vector
            for (size_t k = 0; k < DIM; ++k) {
                yBuf(i, k) = yi[k];
            }
        }
        return y;
    }

    ///< Evaluates the dynamical system for one test point x. Must be implemented by derived classes.
    State eval(const State &x) {
        return static_cast<Derived<dtype, State> *>(this)->_eval(x);
    };

    constexpr std::size_t dim() const { return DIM; }
};

//------------------------------------------------------------------------------
// Virtual base class for ordinary differential equations
//------------------------------------------------------------------------------
template<template<typename, typename> class Derived, typename dtype, typename State>
class ODE : public DynamicalSystemInterface<Derived, dtype, State> {
    using Subclass = Derived<dtype, State>;
public:
    static constexpr std::size_t DIM = std::tuple_size<State>::value;

    State _eval(const State &x) {
        auto h = static_cast<Subclass *>(this)->h();
        auto nSteps = static_cast<Subclass *>(this)->nSteps();
        return _eval(x, h, nSteps);
    }

    State _eval(const State &x, double h, std::size_t nSteps) {
        return integrator.eval([this](const State &_x) {
            return static_cast<Subclass *>(this)->f(_x);
        }, h, nSteps, x);
    }

private:
    deeptime::RungeKutta<State, DIM> integrator;
};

//------------------------------------------------------------------------------
// Virtual base class for stochastic differential equations with constant sigma
//------------------------------------------------------------------------------

template<template<typename, typename> class Derived, typename dtype, typename State>
class SDE : public DynamicalSystemInterface<Derived, dtype, State> {
    using Subclass = Derived<dtype, State>;
public:
    static constexpr std::size_t DIM = std::tuple_size<State>::value;

    State _eval(const State &x) {
        auto h = static_cast<Subclass *>(this)->h();
        auto nSteps = static_cast<Subclass *>(this)->nSteps();
        return _eval(x, h, nSteps);
    }

    State _eval(const State &x, double h, std::size_t nSteps) {
        return _integrator().eval([this](const State &_x) {
            return static_cast<Subclass *>(this)->f(_x);
        }, Subclass::sigma, h, nSteps, x);
    }

private:

    auto &_integrator() {
        return static_cast<Subclass *>(this)->integrator;
    }
};

//------------------------------------------------------------------------------
// ABC flow
//------------------------------------------------------------------------------
template<typename T, typename State = Vector<T, 3>>
class ABCFlow : public ODE<::ABCFlow, T, State> {
public:
    explicit ABCFlow(T h = 1e-3, size_t nSteps = 1000) : _h(h), _nSteps(nSteps) {}

    State f(const State &x) const {
        return {{
            a_ * std::sin(x[2]) + c_ * std::cos(x[1]),
            b_ * std::sin(x[0]) + a_ * std::cos(x[2]),
            c_ * std::sin(x[1]) + b_ * std::cos(x[0])
        }};
    }

    T h() const { return _h; }

    std::size_t nSteps() const { return _nSteps; }

private:
    static constexpr T a_ = 1.73205080757; // sqrt(3)
    static constexpr T b_ = 1.41421356237;  // sqrt(2.);
    static constexpr T c_ = 1;

    T _h;
    std::size_t _nSteps;
};

//------------------------------------------------------------------------------
// Ornstein-Uhlenbeck process
//------------------------------------------------------------------------------
template<typename T, typename State = Vector<T, 1>>
class OrnsteinUhlenbeck : public SDE<OrnsteinUhlenbeck, T, State>, public EulerMaruyamaIntegrator<State, 1, T> {
public:
    static constexpr T alpha = 1.;
    static constexpr T beta = 4.;

    explicit OrnsteinUhlenbeck(std::int64_t seed, double h = 1e-3, size_t nSteps = 500)
            : EulerMaruyamaIntegrator<State, 1, T>(seed), _h(h), _nSteps(nSteps) {}

    State f(const State &x) const {
        return {{ -alpha * x[0] }};
    }

    static constexpr Matrix<T, 1> sigma{{ {{2 / beta}} }};

    T h() const { return _h; }

    std::size_t nSteps() const { return _nSteps; }
private:
    double _h;
    std::size_t _nSteps;
};

//------------------------------------------------------------------------------
// Simple triple-well in one dimension, use interval [0, 6]
//------------------------------------------------------------------------------
template<typename T, typename State = Vector<T, 1>>
class TripleWell1D : public SDE<TripleWell1D, T, State>, public EulerMaruyamaIntegrator<State, 1, T> {
public:
    explicit TripleWell1D(std::int64_t seed, double h = 1e-3, size_t nSteps = 500)
            : EulerMaruyamaIntegrator<State, 1, T>(seed), _h(h), _nSteps(nSteps) {}

    State f(const State &x) const {
        return {{
            -1 * (-24.82002100 + 82.85029600 * x[0] - 82.6031550 * x[0] * x[0]
                       + 34.125104 * std::pow(x[0], 3) - 6.20030 * std::pow(x[0], 4) + 0.4104 * std::pow(x[0], 5))
        }};
    }

    static constexpr Matrix<T, 1> sigma{{{{0.75}}}};

    T h() const { return _h; }

    std::size_t nSteps() const { return _nSteps; }
private:
    double _h;
    std::size_t _nSteps;
};


//------------------------------------------------------------------------------
// Double well problem
//------------------------------------------------------------------------------
template<typename T, typename State = Vector<T, 2>>
class DoubleWell2D : public SDE<DoubleWell2D, T, State>, public EulerMaruyamaIntegrator<State, 2, T> {
public:
    explicit DoubleWell2D(std::int64_t seed, double h = 1e-3, size_t nSteps = 10000)
            : EulerMaruyamaIntegrator<State, 2, T>(seed), _h(h), _nSteps(nSteps) {}

    State f(const State &x) const {
        return {{-4 * x[0] * x[0] * x[0] + 4 * x[0], -2 * x[1]}};
    }

    static constexpr Matrix<T, 2> sigma{{{{0.7, 0.0}}, {{0.0, 0.7}}}};

    T h() const { return _h; }

    std::size_t nSteps() const { return _nSteps; }
private:
    double _h;
    std::size_t _nSteps;
};

//------------------------------------------------------------------------------
// Quadruple well problem
//------------------------------------------------------------------------------
template<typename T, typename State = Vector<T, 2>>
class QuadrupleWell2D : public SDE<QuadrupleWell2D, T, State>, public EulerMaruyamaIntegrator<State, 2, T> {
public:
    explicit QuadrupleWell2D(std::int64_t seed, double h = 1e-3, size_t nSteps = 10000)
            : EulerMaruyamaIntegrator<State, 2, T>(seed), _h(h), _nSteps(nSteps) {}

    State f(const State &x) const {
        // Quadruple well potential: V = (x(1, :).^2 - 1).^2 + (x(2, :).^2 - 1).^2
        return {{-4 * x[0] * x[0] * x[0] + 4 * x[0], -4 * x[1] * x[1] * x[1] + 4 * x[1]}};
    }

    static constexpr T s = 0.70710678118; // sqrt(.5);
    static constexpr Matrix<T, 2> sigma{{{{s, 0.0}}, {{0.0, s}}}};

    T h() const { return _h; }

    std::size_t nSteps() const { return _nSteps; }
private:
    double _h;
    std::size_t _nSteps;
};

//------------------------------------------------------------------------------
// Unsymmetric quadruple well problem
//------------------------------------------------------------------------------
template<typename T, typename State = Vector<T, 2>>
class QuadrupleWellUnsymmetric2D : public SDE<QuadrupleWellUnsymmetric2D, T, State>, public EulerMaruyamaIntegrator<State, 2, T> {
public:
    explicit QuadrupleWellUnsymmetric2D(std::int64_t seed, double h = 1e-3, size_t nSteps = 10000)
            : EulerMaruyamaIntegrator<State, 2, T>(seed), _h(h), _nSteps(nSteps) {}

    State f(const State &x) const {
        return {{
                        -4 * x[0] * x[0] * x[0] + (3.0 / 16.0) * x[0] * x[0] + 4 * x[0] - 3.0 / 16.0,
                        -4 * x[1] * x[1] * x[1] + (3.0 / 8.0) * x[1] * x[1] + 4 * x[1] - 3.0 / 8.0
                }};
    }

    static constexpr Matrix<T, 2> sigma{{{{0.6, 0.0}}, {{0.0, 0.6}}}};

    T h() const { return _h; }

    std::size_t nSteps() const { return _nSteps; }

private:
    double _h;
    std::size_t _nSteps;
};

//------------------------------------------------------------------------------
// Triple well problem
//------------------------------------------------------------------------------
template<typename T, typename State=Vector<T, 2>>
class TripleWell2D : public SDE<TripleWell2D, T, State>, public EulerMaruyamaIntegrator<State, 2, T> {
public:
    explicit TripleWell2D(std::int64_t seed, double h = 1e-5, size_t nSteps = 10000)
            : EulerMaruyamaIntegrator<State, 2, T>(seed), _h(h), _nSteps(nSteps) {}

    State f(const State &x) const {
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

    T h() const { return _h; }

    std::size_t nSteps() const { return _nSteps; }

private:
    double _h;
    std::size_t _nSteps;
};
