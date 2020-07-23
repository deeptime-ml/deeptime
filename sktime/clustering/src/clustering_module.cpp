/********************************************************************************
 * This file is part of scikit-time.                                            *
 *                                                                              *
 * Copyright (c) 2020 AI4Science Group, Freie Universitaet Berlin (GER)         *
 *                                                                              *
 * scikit-time is free software: you can redistribute it and/or modify          *
 * it under the terms of the GNU Lesser General Public License as published by  *
 * the Free Software Foundation, either version 3 of the License, or            *
 * (at your option) any later version.                                          *
 *                                                                              *
 * This program is distributed in the hope that it will be useful,              *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of               *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                *
 * GNU General Public License for more details.                                 *
 *                                                                              *
 * You should have received a copy of the GNU Lesser General Public License     *
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.        *
 ********************************************************************************/

#include "metric.h"
#include "kmeans.h"
#include "regspace.h"

using namespace pybind11::literals;

static const auto euclidean = EuclideanMetric{};

template<typename T>
std::tuple<py::object, int, int, py::object> castLoopResult(const std::tuple<np_array<T>, int, int, np_array<T>> &input) {
    const auto& arr = std::get<0>(input);
    const auto& res = std::get<1>(input);
    const auto& it =  std::get<2>(input);
    const auto& cost = std::get<3>(input);

    return std::make_tuple(py::cast<py::object>(arr), res, it, py::cast<py::object>(cost));
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
            return py::cast<py::object>(std::get<0>(sktime::clustering::kmeans::cluster(py::cast<np_array<float>>(np_chunk),
                                                                    py::cast<np_array<float>>(np_centers),
                                                                    n_threads, metric)));
        } else {
            return py::cast<py::object>(std::get<0>(sktime::clustering::kmeans::cluster(py::cast<np_array<double>>(np_chunk),
                                                                    py::cast<np_array<double>>(np_centers),
                                                                    n_threads, metric)));
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
            auto result = sktime::clustering::kmeans::cluster_loop(
                    py::cast<np_array<float>>(bufChunk), fCenters, metric, n_threads, max_iter,
                    static_cast<float>(tolerance), callback
            );
            return castLoopResult(result);
        } else {
            auto dCenters = py::cast<np_array<double>>(bufCenters);
            auto result = sktime::clustering::kmeans::cluster_loop(
                    py::cast<np_array<double>>(bufChunk), dCenters, metric, n_threads, max_iter,
                    static_cast<double>(tolerance), callback
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
            result = static_cast<double>(sktime::clustering::kmeans::costFunction(
                    py::cast<np_array<float>>(bufChunk), py::cast<np_array<float>>(bufCenters), metric, n_threads));
        } else {
            result = sktime::clustering::kmeans::costFunction(
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
            return py::cast<py::object>(sktime::clustering::kmeans::initKmeansPlusPlus(
                    py::cast<np_array<float>>(bufChunk), k, metric, random_seed, n_threads, callback
                    ));
        } else {
            return py::cast<py::object>(sktime::clustering::kmeans::initKmeansPlusPlus(
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

template<typename dtype, bool squared>
void defDistances(py::module &m) {
    std::string name = "distances";
    if (squared) name += "_squared";
    m.def(name.c_str(), [](np_array<dtype> X, np_array<dtype> Y, py::object XX, py::object YY, int nThreads, const Metric* metric) {
        metric = metric ? metric : &euclidean;
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
        std::size_t nXs = static_cast<std::size_t>(X.shape(0));
        std::size_t nYs = static_cast<std::size_t>(Y.shape(0));

        auto distances = computeDistances<squared>(X.data(), nXs, Y.data(), nYs, dim, xx, yy, metric);
        return distances.numpy();
    }, "X"_a, "Y"_a, "XX"_a = py::none(), "YY"_a = py::none(), "n_threads"_a = 0, "metric"_a = nullptr);
}

PYBIND11_MODULE(_clustering_bindings, m) {
    m.doc() = "module containing clustering algorithms.";
    auto kmeans_mod = m.def_submodule("kmeans");
    registerKmeans(kmeans_mod);
    auto regspace_mod = m.def_submodule("regspace");
    registerRegspace(regspace_mod);

    m.def("assign", [](py::object chunk, py::object centers, int nThreads, const Metric* metric) {
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
    defDistances<float, true>(m);
    defDistances<double, true>(m);
    defDistances<float, false>(m);
    defDistances<double, false>(m);


    py::class_<Metric>(m, "Metric", R"delim(
The metric class. It should not be directly instantiated from python, but is rather meant as a C++ interface. Since
clustering is computationally expensive and the metric is called often, it makes sense to export this functionality
from Python into an extension. To this end the abstract Metric class as defined in `clustering/include/metric.h` can
be implemented and exposed to python. Afterwards it can be used in the clustering module through the
:data:`metric registry <sktime.clustering.metrics>`.
)delim");
    py::class_<EuclideanMetric, Metric>(m, "EuclideanMetric").def(py::init<>());
}
