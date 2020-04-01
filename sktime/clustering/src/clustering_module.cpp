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
    const auto& arr = std::get<0>(input);
    const auto& res = std::get<1>(input);
    const auto& it =  std::get<2>(input);
    const auto& cost = std::get<3>(input);

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
    mod.def("init_centers_kmpp", [](py::object np_data, std::size_t k, std::int64_t random_seed, int n_threads,
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
            clustering::regspace::cluster(py::cast<np_array<double>>(bufChunk), py_centers, dmin,
                                          max_n_clusters, metric ? metric : &euclidean,
                                          n_threads);
        }

    });
    py::register_exception<clustering::regspace::MaxCentersReachedException>(module, "MaxCentersReachedException");
}

PYBIND11_MODULE(_clustering_bindings, m) {
    m.doc() = "module containing clustering algorithms.";
    auto kmeans_mod = m.def_submodule("kmeans");
    registerKmeans(kmeans_mod);
    auto regspace_mod = m.def_submodule("regspace");
    registerRegspace(regspace_mod);

    m.def("assign", [](py::object chunk, py::object centers, std::uint32_t nThreads, const Metric* metric) {
        metric = metric ? metric : &euclidean;

        auto bufChunk = py::array::ensure(chunk);
        auto bufCenters = py::array::ensure(centers);
        if(!(bufChunk && bufCenters)) {
            throw std::invalid_argument("chunk and centers must be numpy arrays.");
        }
        if(py::isinstance<np_array<float>>(bufChunk)) {
            return assign_chunk_to_centers(py::cast<np_array<float>>(bufChunk), py::cast<np_array<float>>(bufCenters),
                                           nThreads, metric);
        } else {
            return assign_chunk_to_centers(py::cast<np_array<double>>(bufChunk), py::cast<np_array<double>>(bufCenters),
                                           nThreads, metric);
        }
    }, "chunk"_a, "centers"_a, "n_threads"_a, "metric"_a = nullptr);

    py::class_<Metric>(m, "Metric", R"delim(
The metric class. It should not be directly instantiated from python, but is rather meant as a C++ interface. Since
clustering is computationally expensive and the metric is called often, it makes sense to export this functionality
from Python into an extension. To this end the abstract Metric class as defined in `clustering/include/metric.h` can
be implemented and exposed to python. Afterwards it can be used in the clustering module.
)delim");
    py::class_<EuclideanMetric, Metric>(m, "EuclideanMetric").def(py::init<>());
}
