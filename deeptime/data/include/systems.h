#pragma once

#include <vector>
#include <random>
#include <chrono>

#include "common.h"
#include "integrator.h"

namespace py = pybind11;

template<class T>
using Vector = std::vector<T>;

template<class T>
using Matrix = std::vector<Vector<T> >;

template<typename T>
class ODE;

template<typename T>
class SDE;


//------------------------------------------------------------------------------
// Virtual base class for all dynamical systems
//------------------------------------------------------------------------------
template<typename T>
class DynamicalSystemInterface {
public:
    virtual ~DynamicalSystemInterface() = default;

    py::array_t<T> operator()(py::array_t<T> x) {
        auto xBuf = x.request();
        T *xPtr = reinterpret_cast<T *>(xBuf.ptr);
        const size_t d = xBuf.shape[0]; // dimension of the state space
        const size_t m = xBuf.shape[1]; // number of snapshots

        py::array_t<T> y = py::array_t<T>(xBuf.size);
        auto yBuf = y.request();
        T *yPtr = reinterpret_cast<T *>(yBuf.ptr);

        Vector<T> xi(d), yi(d);
        for (size_t i = 0; i < m; ++i) // for all test points
        {
            for (size_t k = 0; k < d; ++k) // copy new test point into x vector
                xi[k] = xPtr[k * m + i];

            eval(xi, yi); // evaluate dynamical system

            for (size_t k = 0; k < d; ++k) // copy result into y vector
                yPtr[k * m + i] = yi[k];
        }

        y.resize({d, m});
        return y;
    }

    py::array_t<T> getTrajectory(py::array_t<T> x, size_t length) {
        auto xBuf = x.request();
        T *xPtr = reinterpret_cast<T *>(xBuf.ptr);
        const size_t d = xBuf.shape[0];
        const size_t m = xBuf.shape[1];

        assert(m == 1);

        py::array_t<T> y = py::array_t<T>(d * length);
        auto yBuf = y.request();
        T *yPtr = reinterpret_cast<T *>(yBuf.ptr);

        for (size_t k = 0; k < d; ++k) // copy initial condition
            yPtr[k * length] = xPtr[k];

        Vector<T> xi(d), yi(d);
        for (size_t i = 1; i < length; ++i) {
            for (size_t k = 0; k < d; ++k) // copy new test point into x vector
                xi[k] = yPtr[k * length + (i - 1)];

            eval(xi, yi); // evaluate dynamical system

            for (size_t k = 0; k < d; ++k) // copy result into y vector
                yPtr[k * length + i] = yi[k];
        }

        y.resize({d, length});
        return y;
    }

    virtual void eval(Vector<T> &x,
                      Vector<T> &y) = 0; ///< Evaluates the dynamical system for one test point x. Must be implemented by derived classes.
    virtual size_t getDimension() const = 0; ///< Returns the number of dimensions d of the dynamical system.
};

//------------------------------------------------------------------------------
// Virtual base class for ordinary differential equations
//------------------------------------------------------------------------------
template<typename T>
class ODE : public DynamicalSystemInterface<T> {
public:
    ODE(size_t d, T h, size_t nSteps) : integrator_(this, d, h, nSteps) {}

    // implementation of the pure virtual function inherited from DynamicalSystemInterface
    virtual void eval(Vector<T> &x, Vector<T> &y) {
        integrator_.eval(x, y);
    }

    virtual void f(Vector<T> &x, Vector<T> &y) = 0;

private:
    RungeKutta<T> integrator_;
};

//------------------------------------------------------------------------------
// Virtual base class for stochastic differential equations with constant sigma
//------------------------------------------------------------------------------
template<typename T>
class SDE : public DynamicalSystemInterface<T> {
public:
    SDE(size_t d, T h, size_t nSteps)
            : integrator_(this, d, h, nSteps) {}

    // implementation of the pure virtual function inherited from DynamicalSystemInterface
    void eval(Vector<T> &x, Vector<T> &y) override {
        integrator_.eval(x, y);
    }

    virtual void f(Vector<T> &x, Vector<T> &y) = 0;

    virtual void getSigma(Matrix<T> &sigma) = 0;

private:
    EulerMaruyama<T> integrator_;
};


//------------------------------------------------------------------------------
// ABC flow
//------------------------------------------------------------------------------
template<typename T>
class ABCFlow : public ODE<T> {
public:
    static const size_t d = 3;

    ABCFlow(T h = 1e-3, size_t nSteps = 1000) : ODE<T>(d, h, nSteps), a_(sqrt(3)), b_(sqrt(2)), c_(1) {}

    void f(Vector<T> &x, Vector<T> &y) {
        y[0] = a_ * sin(x[2]) + c_ * cos(x[1]);
        y[1] = b_ * sin(x[0]) + a_ * cos(x[2]);
        y[2] = c_ * sin(x[1]) + b_ * cos(x[0]);
    }

    size_t getDimension() const {
        return d;
    }

private:
    const T a_;
    const T b_;
    const T c_;
};

template<typename T>
void ABCFlowRHS(const T* const x, const T* y) {
    y[0] = a_ * std::sin(x[2]) + c_ * std::cos(x[1]);
    y[1] = b_ * std::sin(x[0]) + a_ * std::cos(x[2]);
    y[2] = c_ * std::sin(x[1]) + b_ * std::cos(x[0]);
}

//------------------------------------------------------------------------------
// Ornstein-Uhlenbeck process
//------------------------------------------------------------------------------
template<typename T>
class OrnsteinUhlenbeck : public SDE<double> {
public:
    static const size_t d = 1;

    OrnsteinUhlenbeck(double h = 1e-3, size_t nSteps = 500)
            : SDE<double>(d, h, nSteps),
              alpha_(1), beta_(4) {}

    void f(Vector<double> &x, Vector<double> &y) {
        y[0] = -alpha_ * x[0];
    }

    void getSigma(Matrix<double> &sigma) {
        sigma[0][0] = sqrt(2 / beta_);
    }

    size_t getDimension() const {
        return d;
    }

private:
    double alpha_;
    double beta_;
};

//------------------------------------------------------------------------------
// Simple triple-well in one dimension, use interval [0, 6]
//------------------------------------------------------------------------------
template<typename T>
class TripleWell1D : public SDE<double> {
public:
    static const size_t d = 1;

    TripleWell1D(double h = 1e-3, size_t nSteps = 500)
            : SDE<double>(d, h, nSteps) {}

    void f(Vector<double> &x, Vector<double> &y) {
        y[0] = -1 * (-24.82002100 + 82.85029600 * x[0] - 82.6031550 * x[0] * x[0]
                     + 34.125104 * std::pow(x[0], 3) - 6.20030 * std::pow(x[0], 4) + 0.4104 * std::pow(x[0], 5));
    }

    void getSigma(Matrix<double> &sigma) {
        sigma[0][0] = 0.75;
    }

    size_t getDimension() const {
        return d;
    }
};

//------------------------------------------------------------------------------
// Double well problem
//------------------------------------------------------------------------------
template<typename T>
class DoubleWell2D : public SDE<double> {
public:
    static const size_t d = 2;

    DoubleWell2D(double h = 1e-3, size_t nSteps = 10000)
            : SDE<double>(d, h, nSteps) {}

    void f(Vector<double> &x, Vector<double> &y) {
        // Double well potential: V = (x(1, :).^2 - 1).^2 + x(2, :).^2
        y[0] = -4 * x[0] * x[0] * x[0] + 4 * x[0];
        y[1] = -2 * x[1];
    }

    void getSigma(Matrix<double> &sigma) {
        sigma[0][0] = 0.7;
        sigma[0][1] = 0.0;
        sigma[1][0] = 0.0;
        sigma[1][1] = 0.7;
    }

    size_t getDimension() const {
        return d;
    }
};

//------------------------------------------------------------------------------
// Quadruple well problem
//------------------------------------------------------------------------------
template<typename T>
class QuadrupleWell2D : public SDE<double> {
public:
    static const size_t d = 2;

    QuadrupleWell2D(double h = 1e-3, size_t nSteps = 10000)
            : SDE<double>(d, h, nSteps) {}

    void f(Vector<double> &x, Vector<double> &y) {
        // Quadruple well potential: V = (x(1, :).^2 - 1).^2 + (x(2, :).^2 - 1).^2
        y[0] = -4 * x[0] * x[0] * x[0] + 4 * x[0];
        y[1] = -4 * x[1] * x[1] * x[1] + 4 * x[1];
    }

    void getSigma(Matrix<double> &sigma) {
        const double s = std::sqrt(0.5);
        sigma[0][0] = s;
        sigma[0][1] = 0;
        sigma[1][0] = 0;
        sigma[1][1] = s;
    }

    size_t getDimension() const {
        return d;
    }

};

//------------------------------------------------------------------------------
// Unsymmetric quadruple well problem
//------------------------------------------------------------------------------
template<typename T>
class QuadrupleWellUnsymmetric2D : public SDE<double> {
public:
    static const size_t d = 2;

    QuadrupleWellUnsymmetric2D(double h = 1e-3, size_t nSteps = 10000)
            : SDE<double>(d, h, nSteps) {}

    void f(Vector<double> &x, Vector<double> &y) {
        y[0] = -4 * x[0] * x[0] * x[0] + (3.0 / 16.0) * x[0] * x[0] + 4 * x[0] - 3.0 / 16.0;
        y[1] = -4 * x[1] * x[1] * x[1] + (3.0 / 8.0) * x[1] * x[1] + 4 * x[1] - 3.0 / 8.0;
    }

    void getSigma(Matrix<double> &sigma) {
        sigma[0][0] = 0.6;
        sigma[0][1] = 0.0;
        sigma[1][0] = 0.0;
        sigma[1][1] = 0.6;
    }

    size_t getDimension() const {
        return d;
    }
};

//------------------------------------------------------------------------------
// Triple well problem
//------------------------------------------------------------------------------
template<typename T>
class TripleWell2D : public SDE<T> {
public:
    static const size_t d = 2;

    TripleWell2D(double h = 1e-5, size_t nSteps = 10000)
            : SDE<T>(d, h, nSteps) {}

    void f(Vector<T> &x, Vector<T> &y) {
        y[0] = -(3 * exp(-x[0] * x[0] - (x[1] - 1.0 / 3) * (x[1] - 1.0 / 3)) * (-2 * x[0])
                 - 3 * exp(-x[0] * x[0] - (x[1] - 5.0 / 3) * (x[1] - 5.0 / 3)) * (-2 * x[0])
                 - 5 * exp(-(x[0] - 1.0) * (x[0] - 1.0) - x[1] * x[1]) * (-2 * (x[0] - 1.0))
                 - 5 * exp(-(x[0] + 1.0) * (x[0] + 1.0) - x[1] * x[1]) * (-2 * (x[0] + 1.0))
                 + 8.0 / 10 * std::pow(x[0], 3));
        y[1] = -(3 * exp(-x[0] * x[0] - (x[1] - 1.0 / 3) * (x[1] - 1.0 / 3)) * (-2 * (x[1] - 1.0 / 3))
                 - 3 * exp(-x[0] * x[0] - (x[1] - 5.0 / 3) * (x[1] - 5.0 / 3)) * (-2 * (x[1] - 5.0 / 3))
                 - 5 * exp(-(x[0] - 1.0) * (x[0] - 1.0) - x[1] * x[1]) * (-2 * x[1])
                 - 5 * exp(-(x[0] + 1.0) * (x[0] + 1.0) - x[1] * x[1]) * (-2 * x[1])
                 + 8.0 / 10 * std::pow(x[1] - 1.0 / 3, 3));
    }

    void getSigma(Matrix<T> &sigma) {
        sigma[0][0] = 1.09;
        sigma[0][1] = 0.0;
        sigma[1][0] = 0.0;
        sigma[1][1] = 1.09;
    }

    size_t getDimension() const {
        return d;
    }
};
