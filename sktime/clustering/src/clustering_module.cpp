//
// Created by marscher on 7/17/17.
//

#include "metric.h"
#include "kmeans.h"
#include "regspace.h"

using namespace pybind11::literals;

static const auto euclidean = EuclideanMetric{};

template<typename T>
std::tuple<py::object, int, int, double> castLoopResult(const std::tuple<np_array<T>, int, int, T> &input) {
    const auto& [arr, res, it, cost] = input;
    return std::make_tuple(py::cast<py::object>(arr), res, it, static_cast<double>(cost));
}

void registerKmeans(py::module &mod) {
    mod.def("cluster", [](py::object np_chunk, py::object np_centers, int n_threads,
                          const Metric *metric) {
        metric = metric ? metric : &euclidean;
        auto bufChunk = py::array::ensure(np_chunk);
        auto bufCenters = py::array::ensure(np_centers);
        if(!(bufChunk && bufCenters)) {
            throw std::invalid_argument("chunk or centers were no numpy arrays.");
        }
        if (py::isinstance<np_array<float>>(bufChunk)) {
            return py::cast<py::object>(clustering::kmeans::cluster(py::cast<np_array<float>>(np_chunk),
                                                                    py::cast<np_array<float>>(np_centers),
                                                                    n_threads, metric));
        } else {
            return py::cast<py::object>(clustering::kmeans::cluster(py::cast<np_array<double>>(np_chunk),
                                                                    py::cast<np_array<double>>(np_centers),
                                                                    n_threads, metric));
        }
    }, "chunk"_a, "centers"_a, "n_threads"_a, "metric"_a = nullptr);
    mod.def("cluster_loop", [](py::object np_chunk, py::object np_centers,
                               std::size_t k, int n_threads, int max_iter, double tolerance,
                               py::object& callback, const Metric *metric) {
        metric = metric ? metric : &euclidean;
        auto bufChunk = py::array::ensure(np_chunk);
        auto bufCenters = py::array::ensure(np_centers);
        if(!(bufChunk && bufCenters)) {
            throw std::invalid_argument("chunk or centers were no numpy arrays.");
        }
        if (py::isinstance<np_array<float>>(bufChunk)) {
            auto fCenters = py::cast<np_array<float>>(bufCenters);
            auto result = clustering::kmeans::cluster_loop(
                    py::cast<np_array<float>>(bufChunk), fCenters, k, metric, n_threads, max_iter,
                    static_cast<float>(tolerance), callback
            );
            return castLoopResult(result);
        } else {
            auto dCenters = py::cast<np_array<double>>(bufCenters);
            auto result = clustering::kmeans::cluster_loop(
                    py::cast<np_array<double>>(bufChunk), dCenters, k, metric, n_threads, max_iter,
                    tolerance, callback
            );
            return castLoopResult(result);
        }
    }, "chunk"_a, "centers"_a, "k"_a, "n_threads"_a, "max_iter"_a, "tolerance"_a, "callback"_a, "metric"_a = nullptr);
    mod.def("cost_function", [](py::object np_data, py::object np_centers, int n_threads,
                                const Metric *metric) {
        metric = metric ? metric : &euclidean;
        auto bufChunk = py::array::ensure(np_data);
        auto bufCenters = py::array::ensure(np_centers);
        if(!(bufChunk && bufCenters)) {
            throw std::invalid_argument("chunk or centers were no numpy arrays.");
        }
        double result;
        if (py::isinstance<np_array<float>>(bufChunk)) {
            result = static_cast<double>(clustering::kmeans::costFunction(
                    py::cast<np_array<float>>(bufChunk), py::cast<np_array<float>>(bufCenters), metric, n_threads));
        } else {
            result = clustering::kmeans::costFunction(
                    py::cast<np_array<double>>(bufChunk), py::cast<np_array<double>>(bufCenters), metric, n_threads);
        }
        return result;
    }, "chunk"_a, "centers"_a, "n threads"_a, "metric"_a = nullptr);
    mod.def("init_centers_kmpp", [](py::object np_data, std::size_t k, unsigned int random_seed, int n_threads,
                                    py::object& callback, const Metric *metric) {
        metric = metric ? metric : &euclidean;
        auto bufChunk = py::array::ensure(np_data);
        if(!bufChunk) {
            throw std::invalid_argument("data was not a numpy array.");
        }
        if(py::isinstance<np_array<float>>(bufChunk)) {
            return py::cast<py::object>(clustering::kmeans::initCentersKMpp(
                    py::cast<np_array<float>>(bufChunk), k, metric, random_seed, n_threads, callback
                    ));
        } else {
            return py::cast<py::object>(clustering::kmeans::initCentersKMpp(
                    py::cast<np_array<double>>(bufChunk), k, metric, random_seed, n_threads, callback
            ));
        }
    }, "chunk"_a, "k"_a, "random_seed"_a, "n_threads"_a, "callback"_a, "metric"_a = nullptr);
}

void registerRegspace(py::module &module) {
    module.def("cluster", [](py::object np_chunk, py::list &py_centers, double dmin,
            std::size_t max_n_clusters, unsigned int n_threads, const Metric *metric) {
        metric = metric ? metric : &euclidean;
        auto bufChunk = py::array::ensure(np_chunk);
        if(!bufChunk) {
            throw std::invalid_argument("data was not a numpy array.");
        }
        if(py::isinstance<np_array<float>>(bufChunk)) {
            clustering::regspace::cluster(py::cast<np_array<float>>(bufChunk), py_centers, static_cast<float>(dmin),
                                          max_n_clusters, metric ? metric : &euclidean,
                                          n_threads);
        } else {
            clustering::regspace::cluster(py::cast<np_array<double>>(bufChunk), py_centers, static_cast<double>(dmin),
                                          max_n_clusters, metric ? metric : &euclidean,
                                          n_threads);
        }

    });
}

PYBIND11_MODULE(_clustering_bindings, m) {
    m.doc() = "module containing clustering algorithms.";
    auto kmeans_mod = m.def_submodule("kmeans");
    registerKmeans(kmeans_mod);
    auto regspace_mod = m.def_submodule("regspace");
    registerRegspace(regspace_mod);

    py::class_<Metric>(m, "Metric");
    py::class_<EuclideanMetric, Metric>(m, "EuclideanMetric").def(py::init<>());
}
