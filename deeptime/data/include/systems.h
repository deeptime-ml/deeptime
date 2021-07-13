#pragma once

#include <vector>
#include <random>
#include <chrono>
#include <complex>

#include "common.h"
#include "integrator.h"
#include "boundary_conditions.h"

namespace py = pybind11;

namespace deeptime::data {

struct ode_tag {
};
struct sde_tag {
};

template<class T, std::size_t DIM>
using Vector = std::array<T, DIM>;

template<typename T, std::size_t DIM>
using Matrix = Vector<Vector<T, DIM>, DIM>;

namespace detail {
template<class, class = void>
struct system_has_potential : std::false_type {
};
template<class T>
struct system_has_potential<T, std::void_t<decltype(std::declval<T>().energy(std::declval<typename T::State>()))>>
        : std::true_type {
};

template<class, class = void>
struct system_has_periodic_boundaries : std::false_type {
};
template<class T>
struct system_has_periodic_boundaries<T, std::void_t<decltype(typename T::Boundary {})>>
        : std::true_type {
};

template<class, class = void>
struct system_has_potential_time : std::false_type {
};
template<class T>
struct system_has_potential_time<T, std::void_t<decltype(std::declval<T>().energy(std::declval<double>(),
                                                                                  std::declval<typename T::State>()))>>
        : std::true_type {
};

}

template<typename T>
static constexpr bool system_has_potential_v =
        detail::system_has_potential<T>::value || detail::system_has_potential_time<T>::value;
template<typename T>
static constexpr bool system_has_periodic_boundaries_v = detail::system_has_periodic_boundaries<T>::value;

//------------------------------------------------------------------------------
// ABC flow
//------------------------------------------------------------------------------
template<typename T>
struct ABCFlow {
    using system_type = ode_tag;

    static constexpr std::size_t DIM = 3;
    using dtype = T;
    using State = Vector<T, DIM>;
    using Integrator = RungeKutta<State, DIM>;

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
// Bickley Jet
//------------------------------------------------------------------------------
template<typename T, bool PERIODIC = false>
struct BickleyJet {
    using system_type = ode_tag;
    static constexpr std::size_t DIM = 2;
    using dtype = T;
    using State = Vector<T, DIM>;
    using vmin_type = std::tuple<std::ratio<0, 1>, std::ratio<-3, 1>>;
    using vmax_type = std::tuple<std::ratio<20, 1>, std::ratio<3, 1>>;
    using Boundary = BoundaryConditions<State, 2, vmin_type, vmax_type, PERIODIC, false>;
    using Integrator = RungeKutta<State, DIM, Boundary>;

    template<typename D>
    static constexpr auto sech(D t) {
        return 1. / std::cosh(t);
    }

    constexpr State f(double t, const State &xVec) const {
        using namespace std::complex_literals;
        auto[x, y] = Boundary::apply(xVec);
        std::complex<T> fc{0};
        std::complex<T> df_dx_c{0};
        for (int j = 0; j < 3; ++j) {
            fc += eps[j] * std::exp(-1i * k[j] * c[j] * t) * std::exp(1i * k[j] * x);
            df_dx_c += eps[j] * std::exp(-1i * k[j] * c[j] * t) * 1i * k[j] * std::exp(1i * k[j] * x);
        }
        auto f = fc.real();
        auto df_dx = df_dx_c.real();
        auto sech_y = sech(y / L0);
        return {{
                        U0 * sech_y * sech_y + 2. * U0 * std::tanh(y / L0) * sech_y * sech_y * f,
                        U0 * L0 * sech_y * sech_y * df_dx
                }};
    }

    static constexpr T U0{5.4138};
    static constexpr T L0{1.77};
    static constexpr T r0{6.371};
    static constexpr std::array<T, 3> eps{{0.075, 0.15, 0.3}};
    static constexpr std::array<T, 3> c{{U0 * 0.1446, U0 * 0.205, U0 * 0.461}};
    static constexpr std::array<T, 3> k{{2. / r0, 4. / r0, 6. / r0}};

    T h{1e-2};
    std::size_t nSteps{static_cast<std::size_t>(0.1 / h)};
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
    using Integrator = EulerMaruyama<State, DIM>;

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
// Prinz potential
//------------------------------------------------------------------------------
template<typename T>
struct Prinz {
    using system_type = sde_tag;

    static constexpr std::size_t DIM = 1;
    using dtype = T;
    using State = Vector<dtype, DIM>;
    using Integrator = EulerMaruyama<State, DIM>;

    constexpr dtype energy(const State &x) const {
        return 4. / (mass * damping) * (std::pow(x[0], 8) + 0.8 * std::exp(-80. * x[0] * x[0])
                                        + 0.2 * std::exp(-80. * (x[0] - .5) * (x[0] - .5))
                                        + 0.5 * std::exp(-40. * (x[0] + .5) * (x[0] + .5)));
    }

    State f(const State &x) const {
        return {{-4. / (mass * damping) * (8. * std::pow(x[0], 7) - 128. * std::exp(-80. * x[0] * x[0]) * x[0]
                                           - 32. * std::exp(-80. * (x[0] - 0.5) * (x[0] - 0.5)) * (x[0] - 0.5)
                                           - 40. * std::exp(-40. * (x[0] + 0.5) * (x[0] + 0.5)) * (x[0] + 0.5))}};
    }

    T h{1e-3};
    std::size_t nSteps{1};
    T mass{1.};
    T damping{1.};
    T kT{1.};

    Matrix<T, 1> sigma{{{{std::sqrt(2. * kT / (mass * damping))}}}};

    void updateSigma() {
        sigma = Matrix<T, 1>{{{{std::sqrt(2. * kT / (mass * damping))}}}};
    };
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
    using Integrator = EulerMaruyama<State, DIM>;

    constexpr dtype energy(const State &x) const {
        return -(24.82 * x[0] - 41.4251 * x[0] * x[0] + 27.5344 * std::pow(x[0], 3)
                 - 8.53128 * std::pow(x[0], 4) + 1.24006 * std::pow(x[0], 5) - 0.0684 * std::pow(x[0], 6)) + 5;
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
    using Integrator = EulerMaruyama<State, DIM>;

    constexpr dtype energy(const State &x) const {
        return (x[0] * x[0] - 1.) * (x[0] * x[0] - 1.) + x[1] * x[1];
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
    using Integrator = EulerMaruyama<State, DIM>;

    constexpr dtype energy(const State &x) const {
        return (x[0] * x[0] - 1) * (x[0] * x[0] - 1) + (x[1] * x[1] - 1) * (x[1] * x[1] - 1);
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
    using Integrator = EulerMaruyama<State, DIM>;

    constexpr dtype energy(const State &x) const {
        return +x[0] * x[0] * x[0] * x[0] - (1. / 16.) * x[0] * x[0] * x[0] - 2. * x[0] * x[0] + (3. / 16.) * x[0]
               + x[1] * x[1] * x[1] * x[1] - (1. / 8.) * x[1] * x[1] * x[1] - 2 * x[1] * x[1] + (3. / 8.) * x[1];
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
    using Integrator = EulerMaruyama<State, DIM>;

    constexpr dtype energy(const State &x) const {
        const auto &xv = x[0];
        const auto &yv = x[1];
        return +3. * std::exp(-(xv * xv) - (yv - 1. / 3.) * (yv - 1. / 3.))
               - 3. * std::exp(-(xv * xv) - (yv - 5. / 3.) * (yv - 5. / 3.))
               - 5. * std::exp(-(xv - 1.) * (xv - 1.) - yv * yv)
               - 5. * std::exp(-(xv + 1.) * (xv + 1.) - yv * yv)
               + (2. / 10.) * xv * xv * xv * xv
               + (2. / 10.) * std::pow(yv - 1. / 3., 4.);
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

//------------------------------------------------------------------------------
// Time-dependent 5-well
//------------------------------------------------------------------------------
template<typename T>
struct TimeDependent5Well {
    using system_type = sde_tag;

    static constexpr std::size_t DIM = 2;
    using dtype = T;
    using State = Vector<T, DIM>;
    using Integrator = EulerMaruyama<State, DIM>;

    constexpr dtype energy(double t, const State &x) const {
        const auto &xv = x[0];
        const auto &yv = x[1];
        auto term1 = std::cos(s * std::atan2(yv, xv) - 0.5 * dt::constants::pi<T>() * t);
        auto term2 = std::sqrt(xv * xv + yv * yv) - 3. / 2 - 0.5 * std::sin(2 * dt::constants::pi<T>() * t);
        return term1 + 10 * term2 * term2;
    }

    constexpr State f(double t, const State &xvec) const {
        auto x = xvec[0];
        auto y = xvec[1];
        auto pi = dt::constants::pi<T>();
        return {{
                        +(s * y * std::sin(0.5 * pi * t - s * std::atan2(y, x)) - 10. * x * std::sqrt(x * x + y * y) *
                                                                                  (-std::sin(2 * pi * t) +
                                                                                   2 * std::sqrt(x * x + y * y) - 3)) /
                        (x * x + y * y),
                        -(s * x * std::sin(0.5 * pi * t - s * std::atan2(y, x)) + 10. * y * std::sqrt(x * x + y * y) *
                                                                                  (-std::sin(2 * pi * t) +
                                                                                   2 * std::sqrt(x * x + y * y) - 3)) /
                        (x * x + y * y)
                }};
    }

    T beta = 5.;
    T h{1e-5};
    T s = 5;
    std::size_t nSteps{10000};

    Matrix<T, 2> sigma{{{{std::sqrt(2. / beta), 0.}}, {{0., std::sqrt(2. / beta)}}}};

    void updateSigma() {
        sigma = {{{{std::sqrt(2. / beta), 0.}}, {{0., std::sqrt(2. / beta)}}}};
    };
};

template<typename T, typename = void>
struct is_time_dependent : std::false_type {
};

template<typename T>
struct is_time_dependent<T, std::void_t<decltype(std::declval<T>().f(0., typename T::State{}))>> : std::true_type {
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
                                double t0, const typename System::State &x, double h, std::size_t nSteps, ode_tag) {
    if constexpr(is_time_dependent<System>::value) {
        auto rhs = [&](typename System::dtype t, const auto &_x) {
            return system.f(t, _x);
        };
        return integrator.eval(rhs, h, nSteps, t0, x);
    } else {
        auto rhs = [&](typename System::dtype, const auto &_x) {
            return system.f(_x);
        };
        return integrator.eval(rhs, h, nSteps, t0, x);
    }
}

template<typename System>
typename System::State evaluate(const System &system, typename System::Integrator &integrator,
                                double t0, const typename System::State &x, double h, std::size_t nSteps, sde_tag) {
    if constexpr(is_time_dependent<System>::value) {
        auto rhs = [&system](typename System::dtype t, const auto &_x) {
            return system.f(t, _x);
        };
        if constexpr(detail::IsMemberSigma<System>{}) {
            return integrator.eval(rhs, system.sigma, h, nSteps, t0, x);
        } else {
            return integrator.eval(rhs, System::sigma, h, nSteps, t0, x);
        }
    } else {
        auto rhs = [&system](typename System::dtype, const auto &_x) {
            return system.f(_x);
        };
        if constexpr(detail::IsMemberSigma<System>{}) {
            return integrator.eval(rhs, system.sigma, h, nSteps, t0, x);
        } else {
            return integrator.eval(rhs, System::sigma, h, nSteps, t0, x);
        }
    }
}
}


template<typename System>
typename System::State evaluate(const System &system, typename System::Integrator &integrator,
                                double t0, const typename System::State &x, double h, std::size_t nSteps) {
    return detail::evaluate(system, integrator, t0, x, h, nSteps, typename System::system_type());
}

template<typename System>
typename System::State evaluate(const System &system, typename System::Integrator &integrator,
                                double t0, const typename System::State &x) {
    return detail::evaluate(system, integrator, t0, x, system.h, system.nSteps, typename System::system_type());
}

template<typename System>
typename System::Integrator createIntegrator(std::int64_t /*seed*/, ode_tag) {
    return typename System::Integrator{};
}

template<typename System>
typename System::Integrator createIntegrator(std::int64_t seed, sde_tag) {
    return typename System::Integrator{seed};
}

namespace detail {
template<typename Time, typename System>
auto toBuf(const Time &arr, System) {
    return arr.template unchecked<1>();
}

template<typename System>
auto toBuf(const double &t0, const System &system) {
    auto h = system.h;
    auto nSteps = system.nSteps;
    return [t0, h, nSteps](int i) {
        return t0 + h * nSteps * i;
    };
}

template<typename Time>
auto toOuterBuf(const Time &arr) {
    return arr.template unchecked<1>();
}

template<>
auto toOuterBuf(const double &t0) {
    return [t0](int) {
        return t0;
    };
}
}

template<typename dtype, typename System, typename Time>
np_array_nfc<dtype> evaluateSystem(const System &system, const Time &tArr, const np_array_nfc<dtype> &x,
                                   std::int64_t seed = -1, int nThreads = -1) {
    if (seed >= 0 && nThreads != 1) {
        throw std::invalid_argument("Fixing the seed requires setting the number of threads to 1.");
    }
    np_array_nfc<dtype> y({x.shape(0), x.shape(1)});

    auto xBuf = x.template unchecked<2>();
    auto tBuf = detail::toBuf(tArr, system);
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
    #pragma omp parallel default(none) firstprivate(system, nTestPoints, xBuf, yBuf, tBuf, testPoint, seed)
    {
        auto integrator = createIntegrator<System>(seed, typename System::system_type());

        #pragma omp for
        for (py::ssize_t i = 0; i < nTestPoints; ++i) {

            // copy new test point into x vector
            for (std::size_t k = 0; k < System::DIM; ++k) {
                testPoint[k] = xBuf(i, k);
            }
            auto t = tBuf(i);
            auto yi = evaluate(system, integrator, t, testPoint); // evaluate dynamical system

            for (std::size_t k = 0; k < System::DIM; ++k) {
                // copy result into y vector
                yBuf(i, k) = yi[k];
            }
        }
    }

    return y;
}

template<typename dtype, typename System, typename Time>
auto
trajectory(System &system, const Time &tArr, const np_array_nfc<dtype> &x, std::size_t length, std::int64_t seed = -1,
           int nThreads = 1) {
    {
        const auto d = x.shape(1);
        if (d != System::DIM) {
            std::stringstream errmsg;
            errmsg << "The dynamical system is " << System::DIM << "-dimensional, but the provided test points had "
                   << d << " dimensions.";
            throw std::invalid_argument(errmsg.str());
        }
    }
    std::size_t nTestPoints = x.shape(0);

    if (nTestPoints > 1 && seed >= 0 && nThreads != 1) {
        throw std::invalid_argument(
                "Fixing the seed for multiple test points requires setting the number of threads to 1.");
    }

    #if defined(USE_OPENMP)
    if (nThreads > 0) {
        omp_set_num_threads(nThreads);
    }
    #else
    nThreads = 0;
    #endif

    np_array_nfc<dtype> y({nTestPoints, length, System::DIM});
    np_array_nfc<double> tOut({nTestPoints, length});

    auto xBuf = x.template unchecked<2>();
    auto yBuf = y.template mutable_unchecked<3>();
    auto tOutBuf = tOut.template mutable_unchecked<2>();
    auto tBufOuter = detail::toOuterBuf(tArr);

    #pragma omp parallel default(none) firstprivate(system, nTestPoints, xBuf, yBuf, tOutBuf, tBufOuter, seed, length)
    {
        auto integrator = createIntegrator<System>(seed, typename System::system_type());

        #pragma omp for
        for (std::size_t testPointIndex = 0; testPointIndex < nTestPoints; ++testPointIndex) {
            auto tBuf = detail::toBuf(tBufOuter(testPointIndex), system);
            for (size_t k = 0; k < System::DIM; ++k) {
                // copy initial condition
                yBuf(testPointIndex, 0, k) = xBuf(testPointIndex, k);
            }

            auto tEval = tBuf(0);
            tOutBuf(testPointIndex, 0) = tEval;

            typename System::State testPoint;
            for (size_t i = 1; i < length; ++i) {
                for (size_t k = 0; k < System::DIM; ++k) {
                    // copy new test point into x vector
                    testPoint[k] = yBuf(testPointIndex, i - 1, k);
                }

                // evaluate dynamical system
                auto yi = evaluate(system, integrator, tEval, testPoint);

                // copy result into y vector
                for (size_t k = 0; k < System::DIM; ++k) {
                    yBuf(testPointIndex, i, k) = yi[k];
                }
                tEval = tBuf(i);
                tOutBuf(testPointIndex, i) = tEval;
            }
        }

    }

    return std::make_tuple(tOut, y);
}
}
