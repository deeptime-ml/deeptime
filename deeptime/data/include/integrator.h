//
// Created by mho on 11/20/20.
//

#pragma once

#include "common.h"
#include "distribution_utils.h"
#include "boundary_conditions.h"

namespace deeptime::data {

//------------------------------------------------------------------------------
// Runge-Kutta integrator for ordinary differential equations
//------------------------------------------------------------------------------
template<typename State, std::size_t DIM, typename Boundary = NoPeriodicBoundaryConditions<State>>
class RungeKutta {
public:
    static constexpr const char* name = "RungeKutta";

    template<typename F>
    State eval(F &&f, double h, std::size_t nSteps, double t0, const State &y0) {
        auto y = y0;
        auto t = t0;
        for (std::size_t i = 0; i < nSteps; ++i) {
            y = step(f, h, t, y);
            t += h;
        }

        return y;
    }

    template<typename F>
    State step(F &&f, double h, double t, const State &y) {
        State yt;

        k1 = f(t, y);  // k1 = f(t, y)
        for (std::size_t j = 0; j < DIM; ++j) {
            yt[j] = y[j] + (h / 2.) * k1[j];
        }
        k2 = f(t + h / 2., yt); // compute k2 = f(t + h/2, y+h/2*k1)
        for (std::size_t j = 0; j < DIM; ++j) {
            yt[j] = y[j] + (h / 2.) * k2[j];
        }
        k3 = f(t + h / 2., yt); // compute k3 = f(t + h/2, y+h/2*k2)
        for (std::size_t j = 0; j < DIM; ++j) {
            yt[j] = y[j] + h * k3[j];
        }
        k4 = f(t + h, yt);  // compute k4 = f(t+h, y+h*k3)

        for (size_t j = 0; j < DIM; ++j) {
            // compute x_{k+1} = x_k + h/6*(k1 + 2*k2 + 2*k3 + k4)
            yt[j] = y[j] + (h / 6.0) * (k1[j] + 2. * k2[j] + 2. * k3[j] + k4[j]);
        }

        return Boundary::apply(yt);
    }

private:
    State k1, k2, k3, k4;
};

//------------------------------------------------------------------------------
// Euler-Maruyama integrator for stochastic differential equations
//------------------------------------------------------------------------------

template<typename State, std::size_t DIM, typename Value = double, typename Generator = std::mt19937>
class EulerMaruyama {
public:

    static constexpr const char* name = "EulerMaruyama";

    explicit EulerMaruyama(std::int64_t seed = -1) {
        generator = seed < 0 ? rnd::randomlySeededGenerator<Generator>() : rnd::seededGenerator<Generator>(seed);
    }

    template<typename F, typename Sigma>
    State eval(F &&f, const Sigma& sigma, double h, std::size_t nSteps, double t0, const State &y0) {
        auto y = y0;
        auto t = t0;
        auto sqrth = std::sqrt(h);
        for (std::size_t i = 0; i < nSteps; ++i) {
            y = step(f, h, sqrth, t, y, sigma);
            t += h;
        }

        return y;
    }

    template<typename F, typename Sigma>
    State step(F &&f, double h, double sqrth, double t, const State &y, const Sigma &sigma) {
        auto mu = f(t, y);
        auto w = noise(); // evaluate Wiener processes
        auto out = y;

        for (size_t j = 0; j < DIM; ++j) {
            out[j] += h * mu[j];

            for (size_t k = 0; k < DIM; ++k) {
                out[j] += sigma[j][k] * sqrth * w[k];
            }
        }
        return out;
    }

private:

    State noise() {
        State out;
        std::generate(out.begin(), out.end(), [this](){
            return distribution(generator);
        });
        return out;
    }

    Generator generator;       ///< random number generator
    std::normal_distribution<Value> distribution;   ///< normal distribution
};
}
