//
// Created by marscher on 7/17/17.
//

#include "metric_base.h"
#include "kmeans.h"
#include "regspace.h"

template<typename T>
using UserMetric = std::function<T(const T*, const T*, std::size_t)>;

template<typename T, typename Module>
void registerKmeans(Module &mod) {
    mod.def("cluster", [](const np_array<T> &np_chunk, const np_array<T> &np_centers, int n_threads) {
        return clustering::kmeans::cluster(np_chunk, np_centers, n_threads, metric::euclidean<T>);
    });
    mod.def("cluster_userdefined_metric", [](const np_array<T> &np_chunk, const np_array<T> &np_centers,
                                             int n_threads, const UserMetric<T> &metric) {
        return clustering::kmeans::cluster(np_chunk, np_centers, n_threads, metric);
    });
    mod.def("cluster_loop", [](const np_array<T>& np_chunk, np_array<T>& np_centers,
                               std::size_t k, int n_threads, int max_iter, T tolerance,
                               py::object& callback) {
        return clustering::kmeans::cluster_loop(np_chunk, np_centers, k, metric::euclidean<T>, n_threads, max_iter,
                                                tolerance, callback);
    });
    mod.def("cluster_loop_userdefined_metric", [](const np_array<T>& np_chunk, np_array<T>& np_centers,
                                                  std::size_t k, int n_threads, int max_iter, T tolerance,
                                                  const UserMetric<T> &metric, py::object& callback) {
        return clustering::kmeans::cluster_loop(np_chunk, np_centers, k, metric, n_threads, max_iter,
                                                tolerance, callback);
    });
    mod.def("init_centers_kmpp", [](const np_array<T>& np_data, std::size_t k, unsigned int random_seed, int n_threads,
                                    py::object& callback) {
        return clustering::kmeans::initCentersKMpp(np_data, k, metric::euclidean<T>, random_seed, n_threads, callback);
    });
    mod.def("init_centers_kmpp_userdefined_metric", [](const np_array<T>& np_data, std::size_t k,
                                                       unsigned int random_seed, int n_threads,
                                                       const UserMetric<T> &metric, py::object& callback) {
        return clustering::kmeans::initCentersKMpp(np_data, k, metric, random_seed, n_threads, callback);
    });
}

template<typename T, typename Module>
void registerRegspace(Module &mod) {
    mod.def("cluster", [](const np_array<T> &np_chunk, py::list &py_centers, T dmin,
                          std::size_t max_n_clusters, unsigned int n_threads) {
        clustering::regspace::cluster(np_chunk, py_centers, dmin, max_n_clusters, metric::euclidean<T>, n_threads);
    });
    mod.def("cluster_userdefined_metric", [](const np_array<T> &np_chunk, py::list &py_centers, T dmin,
                                             std::size_t max_n_clusters, unsigned int n_threads,
                                             const UserMetric<T> &metric) {
        clustering::regspace::cluster(np_chunk, py_centers, dmin, max_n_clusters, metric, n_threads);
    });
}

PYBIND11_MODULE(_ext, m) {
    m.doc() = "module containing clustering algorithms.";
    auto kmeans_mod = m.def_submodule("kmeans");
    registerKmeans<double>(kmeans_mod);
    registerKmeans<float>(kmeans_mod);
    auto regspace_mod = m.def_submodule("regspace");
    registerRegspace<double>(regspace_mod);
    registerRegspace<float>(regspace_mod);
}
