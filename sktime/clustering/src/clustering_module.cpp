//
// Created by marscher on 7/17/17.
//

#include "metric.h"
#include "kmeans.h"
#include "regspace.h"

using namespace pybind11::literals;

static const auto euclidean = EuclideanMetric{};

template<typename T, typename Module>
void registerKmeans(Module &mod) {
    mod.def("cluster", [](const np_array<T> &np_chunk, const np_array<T> &np_centers, int n_threads,
                          const Metric *metric) {
        return clustering::kmeans::cluster(np_chunk, np_centers, n_threads, metric ? metric : &euclidean);
    }, "chunk"_a, "centers"_a, "n_threads"_a, "metric"_a = nullptr);
    mod.def("cluster_loop", [](const np_array<T>& np_chunk, np_array<T>& np_centers,
                               std::size_t k, int n_threads, int max_iter, T tolerance,
                               py::object& callback, const Metric *metric) {
        return clustering::kmeans::cluster_loop(np_chunk, np_centers, k, metric ? metric : &euclidean,
                                                n_threads, max_iter, tolerance, callback);
    }, "chunk"_a, "centers"_a, "k"_a, "n_threads"_a, "max_iter"_a, "tolerance"_a, "callback"_a, "metric"_a = nullptr);
    mod.def("cost_function", [](const np_array<T>& np_data, const np_array<T>& np_centers, int n_threads,
                                const Metric *metric) {
        return clustering::kmeans::costFunction(np_data, np_centers, metric ? metric : &euclidean, n_threads);
    }, "chunk"_a, "centers"_a, "n threads"_a, "metric"_a = nullptr);
    mod.def("init_centers_kmpp", [](const np_array<T>& np_data, std::size_t k, unsigned int random_seed, int n_threads,
                                    py::object& callback, const Metric *metric) {
        return clustering::kmeans::initCentersKMpp(np_data, k, metric ? metric : &euclidean, random_seed,
                                                   n_threads, callback);
    }, "chunk"_a, "k"_a, "random_seed"_a, "n_threads"_a, "callback"_a, "metric"_a = nullptr);
}

template<typename T, typename Module>
void registerRegspace(Module &mod) {
    mod.def("cluster", [](const np_array<T> &np_chunk, py::list &py_centers, T dmin,
                          std::size_t max_n_clusters, unsigned int n_threads, const Metric *metric) {
        clustering::regspace::cluster(np_chunk, py_centers, dmin, max_n_clusters, metric ? metric : &euclidean,
                                      n_threads);
    });
}

PYBIND11_MODULE(_bindings, m) {
    m.doc() = "module containing clustering algorithms.";
    auto kmeans_mod = m.def_submodule("kmeans");
    registerKmeans<double>(kmeans_mod);
    registerKmeans<float>(kmeans_mod);
    auto regspace_mod = m.def_submodule("regspace");
    registerRegspace<double>(regspace_mod);
    registerRegspace<float>(regspace_mod);

    py::class_<Metric>(m, "Metric");
}
