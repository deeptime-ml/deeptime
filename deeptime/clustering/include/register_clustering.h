#pragma once

#include "metric.h"
#include "kmeans.h"
#include "regspace.h"

namespace deeptime{
namespace clustering {

template<typename Metric>
void registerKmeans(py::module &mod) {
    using namespace pybind11::literals;
    mod.def("cluster", kmeans::cluster<Metric, float>, "chunk"_a, "centers"_a,
            "n_threads"_a);
    mod.def("cluster", kmeans::cluster<Metric, double>, "chunk"_a, "centers"_a,
            "n_threads"_a);
    mod.def("cluster_loop", &kmeans::cluster_loop<Metric, float>,
            "chunk"_a, "centers"_a, "n_threads"_a, "max_iter"_a, "tolerance"_a,
            "callback"_a);
    mod.def("cluster_loop", &kmeans::cluster_loop<Metric, double>,
            "chunk"_a, "centers"_a, "n_threads"_a, "max_iter"_a, "tolerance"_a,
            "callback"_a);
    mod.def("cost_function", &kmeans::costAssignFunction<Metric, float>,
            "chunk"_a, "centers"_a, "n_threads"_a);
    mod.def("cost_function", &kmeans::costAssignFunction<Metric, double>,
            "chunk"_a, "centers"_a, "n_threads"_a);
    mod.def("init_centers_kmpp", &kmeans::initKmeansPlusPlus<Metric, float>,
            "chunk"_a, "k"_a, "random_seed"_a, "n_threads"_a, "callback"_a);
    mod.def("init_centers_kmpp", &kmeans::initKmeansPlusPlus<Metric, double>,
            "chunk"_a, "k"_a, "random_seed"_a, "n_threads"_a, "callback"_a);
}

template<typename Metric>
void registerRegspace(py::module &module) {
    using namespace pybind11::literals;
    module.def("cluster", &regspace::cluster<Metric, float>, "chunk"_a, "centers"_a, "dmin"_a,
               "max_clusters"_a, "n_threads"_a);
    module.def("cluster", &regspace::cluster<Metric, double>, "chunk"_a, "centers"_a, "dmin"_a,
               "max_clusters"_a, "n_threads"_a);
    py::register_exception<regspace::MaxCentersReachedException>(module, "MaxCentersReachedException");
}

template<typename dtype, bool squared, typename Metric>
void defDistances(py::module &m) {
    using namespace pybind11::literals;
    std::string name = "distances";
    if (squared) name += "_squared";
    m.def(name.c_str(), [](np_array<dtype> X, np_array<dtype> Y, py::object XX, py::object YY, int /*nThreads*/) {
        auto dim = static_cast<std::size_t>(X.shape(1));
        if(static_cast<std::size_t>(Y.shape(1)) != dim) {
            throw std::invalid_argument("dimension mismatch: " + std::to_string(dim) + " != " + std::to_string(Y.shape(1)));
        }
        const double* xx = nullptr;
        if(!XX.is_none()) {
            xx = py::cast<np_array<double>>(XX).data();
        }
        const double* yy = nullptr;
        if(!YY.is_none()) {
            yy = py::cast<np_array<double>>(YY).data();
        }
        auto nXs = static_cast<std::size_t>(X.shape(0));
        auto nYs = static_cast<std::size_t>(Y.shape(0));

        auto distances = computeDistances<squared, Metric>(X.data(), nXs, Y.data(), nYs, dim, xx, yy);
        return distances.numpy();
    }, "X"_a, "Y"_a, "XX"_a = py::none(), "YY"_a = py::none(), "n_threads"_a = 0);
}

template<typename Metric>
void registerAssignFunctions(py::module &module) {
    using namespace pybind11::literals;
    module.def("assign", &assign_chunk_to_centers<Metric, float>, "chunk"_a, "centers"_a, "n_threads"_a);
    module.def("assign", &assign_chunk_to_centers<Metric, double>, "chunk"_a, "centers"_a, "n_threads"_a);

    defDistances<float, true, Metric>(module);
    defDistances<double, true, Metric>(module);
    defDistances<float, false, Metric>(module);
    defDistances<double, false, Metric>(module);
}

template<typename Metric>
void registerClusteringImplementation(py::module &m) {
    auto kmeans = m.def_submodule("kmeans");
    registerKmeans<Metric>(kmeans);
    auto regspace = m.def_submodule("regspace");
    registerRegspace<Metric>(regspace);

    registerAssignFunctions<Metric>(m);
}

}
}
