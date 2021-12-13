//
// Created by mho on 12/13/21.
//

#pragma once

namespace deeptime::data {

template<class, class>
struct system_has_potential : std::false_type {};
template<class T>
struct system_has_potential<T, std::void_t<decltype(std::declval<T>().energy(std::declval<typename T::State>()))>>
        : std::true_type { };

template<class, class>
struct system_has_potential_time : std::false_type { };
template<class T>
struct system_has_potential_time<T, std::void_t<decltype(std::declval<T>().energy(std::declval<double>(),
                                                                                  std::declval<typename T::State>()))>>
        : std::true_type { };

template<class, class>
struct system_has_periodic_boundaries : std::false_type { };
template<class T>
struct system_has_periodic_boundaries<T, std::void_t<decltype(typename T::Boundary {})>> : std::true_type { };

template<typename T, typename>
struct is_time_dependent : std::false_type {};
template<typename T>
struct is_time_dependent<T, std::void_t<decltype(std::declval<T>().f(0., typename T::State{}))>> : std::true_type {};

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
    if constexpr(is_time_dependent_v<System>) {
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
    if constexpr(is_time_dependent_v<System>) {
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

template<typename System>
inline typename System::State evaluate(const System &system, typename System::Integrator &integrator,
                                       double t0, const typename System::State &x, double h, std::size_t nSteps) {
    return detail::evaluate(system, integrator, t0, x, h, nSteps, typename System::system_type());
}
template<typename System>
inline typename System::State evaluate(const System &system, typename System::Integrator &integrator,
                                       double t0, const typename System::State &x) {
    return detail::evaluate(system, integrator, t0, x, system.h, system.nSteps, typename System::system_type());
}

template<typename dtype, typename System, typename Time>
inline np_array_nfc<dtype> evaluateSystem(const System &system, const Time &tArr, const np_array_nfc<dtype> &x,
                                          std::int64_t seed, int nThreads) {
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
inline auto trajectory(System &system, const Time &tArr, const np_array_nfc<dtype> &x, std::size_t length,
                       std::int64_t seed, int nThreads) {
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
